from mpi4py import MPI
from dolfinx import mesh, fem, io, plot, nls, log, geometry, la
from dolfinx import cpp as _cpp
from dolfinx.nls.petsc import NewtonSolver
from dolfinx.fem.forms import extract_function_spaces
import dolfinx.fem.petsc
import ufl
import numpy as np
from petsc4py.PETSc import ScalarType
from petsc4py import PETSc
import time
import os




# Material properties
E, nu = ScalarType(70000), ScalarType(0.22)	                                    #Young's modulus and Poisson's ratio
mu, lmbda, kappa = E/(2*(1 + nu)), E*nu/((1 + nu)*(1 - 2*nu)), E/(3*(1 - 2*nu))
Gc= ScalarType(0.010)	                                                          #Critical energy release rate
sts, scs= ScalarType(40), ScalarType(1000)	                                      #Tensile strength and compressive strength
shs = (2/3)*sts*scs/(scs-sts)
Wts = sts**2/(2*E)
Whs = shs**2/(2*kappa)



#The regularization length
eps = 0.016
h = 0.005



delta = (1+3*h/(8*eps))**(-2) * ((sts + (1+2*np.sqrt(3))*shs)/((8+3*np.sqrt(3))*shs)) * 3*Gc/(16*Wts*eps) + (1+3*h/(8*eps))**(-1) * (2/5)
comm = MPI.COMM_WORLD
comm_rank = MPI.COMM_WORLD.rank
log.set_log_level(log.LogLevel.ERROR)

#Geometry of the specimen
W, L = 25.0, 25.0


# check if Inden2D.msh exists
if not os.path.exists("Inden2D.msh"):
    raise FileNotFoundError("Inden2D.msh not found. Please download the mesh from the nine circles of elastic brittle fracture repository.")

partitioner = dolfinx.cpp.mesh.create_cell_partitioner(mesh.GhostMode.shared_facet, None)
mesh_data = io.gmsh.read_from_msh("Inden2D.msh", MPI.COMM_WORLD, gdim=2, partitioner=partitioner)
domain = mesh_data[0]






domain.topology.create_connectivity(domain.topology.dim, domain.topology.dim)
domain.topology.create_connectivity(domain.topology.dim-1, domain.topology.dim)
domain.topology.create_connectivity(domain.topology.dim, domain.topology.dim-1)
domain.topology.create_connectivity(domain.topology.dim-1, domain.topology.dim-1)
domain.topology.create_connectivity(0, domain.topology.dim)


# Defining the function spaces
V = fem.functionspace(domain, ("CG", 1, (domain.geometry.dim,)))                  #Function space for u
Y = fem.functionspace(domain, ("CG", 1))                                          #Function space for z

maxdisp = -0.02
indentor_radius = 1

def left(x):
    return np.isclose(x[0], 0)

def front(x):
    return np.isclose(x[0], W)

def top(x):
    return np.isclose(x[1], 0)

def bottom(x):
    return np.isclose(x[1], -L)

def loadset(x):
    return (x[0] > -1e-4) & (x[0]< indentor_radius + 1e-4) & (abs(x[1]-0) < 1e-4)

def outer(x):
    return (x[1] < -L/10)

fdim = domain.topology.dim -1
left_facets = mesh.locate_entities_boundary(domain, fdim, left)
front_facets = mesh.locate_entities_boundary(domain, fdim, front)
top_facets = mesh.locate_entities_boundary(domain, fdim, top)
bottom_facets = mesh.locate_entities(domain, fdim, bottom)
loadset_facets = mesh.locate_entities(domain, fdim, loadset)
outer_facets = mesh.locate_entities(domain, fdim, outer)


dofs_left = fem.locate_dofs_topological(V.sub(0), fdim, left_facets)
dofs_bottom = fem.locate_dofs_topological(V.sub(1), fdim, bottom_facets)
dofs_loadset = fem.locate_dofs_topological(V.sub(1), fdim, loadset_facets)

dofs_outer = fem.locate_dofs_topological(Y, fdim, outer_facets)

bcl = fem.dirichletbc(ScalarType(0), dofs_left, V.sub(0))
bcb = fem.dirichletbc(ScalarType(0), dofs_bottom, V.sub(1))
bct = fem.dirichletbc(ScalarType(0), dofs_loadset, V.sub(1))
bcs = [bcl, bcb, bct]


bcs_z = []



marked_facets = np.hstack([top_facets, bottom_facets, left_facets])
marked_values = np.hstack([np.full_like(top_facets, 1),
                           np.full_like(bottom_facets, 2),
                           np.full_like(left_facets, 3)])
sorted_facets = np.argsort(marked_facets)
facet_tag = mesh.meshtags(domain, domain.topology.dim -1,
                          marked_facets[sorted_facets],
                          marked_values[sorted_facets])

metadata = {"quadrature_degree": 2}
ds = ufl.Measure('ds', domain=domain,
                 subdomain_data=facet_tag, metadata=metadata)
dS = ufl.Measure("dS", domain=domain, metadata=metadata)
dx = ufl.Measure("dx", domain=domain, metadata=metadata)

xh = ufl.SpatialCoordinate(domain) 

# Define functions
du = ufl.TrialFunction(V)                                                         # Incremental displacement
v  = ufl.TestFunction(V)                                                          # Test function for u
u  = fem.Function(V, name="displacement")                                         # Displacement from previous iteration
u_inc = fem.Function(V)
dz = ufl.TrialFunction(Y)                                                         # Incremental phase field
y  = ufl.TestFunction(Y)                                                          # Test function for z
z  = fem.Function(Y, name="phasefield")                                           # Phase field from previous iteration
z_inc = fem.Function(Y)
d = len(u)


u.x.array[:] = 0.


z.x.array[:] = 1.


u_prev = fem.Function(V)
u_prev.x.array[:] = u.x.array
z_prev = fem.Function(Y)
z_prev.x.array[:] = z.x.array


y_dofs_top = fem.locate_dofs_topological(V.sub(1), fdim, loadset_facets)


def adjust_array_shape(input_array):
    if input_array.shape == (2,):                                                 # Check if the shape is (2,)
        adjusted_array = np.append(input_array, 0.0)                              # Append 0.0 to the array
        return adjusted_array
    else:
        return input_array
bb_tree = geometry.bb_tree(domain, domain.topology.dim)

def evaluate_function(u, x):
    """[summary]
        Helps evaluated a function at a point `x` in parallel
    Args:
        u ([dolfin.Function]): [function to be evaluated]
        x ([Union(tuple, list, numpy.ndarray)]): [point at which to evaluate function `u`]

    Returns:
        [numpy.ndarray]: [function evaluated at point `x`]
    """


    if isinstance(x, np.ndarray):
        # If x is already a NumPy array
        points0 = x
    elif isinstance(x, (tuple, list)):
        # If x is a tuple or list, convert it to a NumPy array
        points0 = np.array(x)
    else:
        # Handle the case if x is of an unsupported type
        points0 = None

    points = adjust_array_shape(points0)

    u_value = []

    cells = []
    # Find cells whose bounding-box collide with the the points
    cell_candidates = geometry.compute_collisions_points(bb_tree, points)
    # Choose one of the cells that contains the point
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points)

    if len(colliding_cells.links(0)) > 0:
        u_value = u.eval(points, colliding_cells.links(0)[0])
        u_value = domain.comm.gather(u_value, root=0)
    return u_value[0]

# Stored energy, strain and stress functions in linear isotropic elasticity (axisymmetric)

def energy(v):
    return mu*(ufl.inner(epsilon(v),epsilon(v))) + 0.5*(lmbda)*(ufl.tr(epsilon(v)))**2


def epsilon(v):
    return ufl.sym(
        ufl.as_tensor(
            [
                [v[0].dx(0), 0, v[0].dx(1)],
                [0, v[0] / xh[0], 0],
                [v[1].dx(0), 0, v[1].dx(1)],
            ]
        )
    )

def sigma(v):
    return 2.0*mu*epsilon(v) + (lmbda)*ufl.tr(epsilon(v))*ufl.Identity(3)


def epsilonD(v):
    return epsilon(v) - (1/3)*ufl.tr(epsilon(v))*ufl.Identity(3)

def sigmaD(v):
    return 2.0*mu*epsilonD(v)
 
def sigmavm(v):
    return ufl.sqrt(1/2*(ufl.inner(sigmaD(v),sigmaD(v))))

eta = 0.0
# Stored energy density
psi1 = (z**2+eta)*(energy(u))
psi11 = energy(u)
# Total potential energy
Pi = psi1*xh[0]*dx
# Compute first variation of Pi (directional derivative about u in the direction of v)
R = ufl.derivative(Pi, u, v)
# Compute Jacobian of R
Jac = ufl.derivative(R, u, du)


I1 = (z**2)*ufl.tr(sigma(u))
SQJ2 = (z**2)*sigmavm(u)

alpha1 = (delta*Gc)/(shs*8*eps) - (2*Whs)/(3*shs)
alpha2 = (3**0.5*(3*shs - sts)*delta*Gc)/(shs*sts*8*eps) + (2*Whs)/(3**0.5*shs) - (2*3**0.5*Wts)/(sts)

ce = alpha2*SQJ2 + alpha1*I1 - z*(1-ufl.sqrt(I1**2)/I1)*psi11


#Balance of configurational forces PDE
pen=1000*(3*Gc/8/eps)*ufl.conditional(ufl.lt(delta,1),1, delta)
Wv=pen/2*((abs(z)-z)**2 + (abs(1-z) - (1-z))**2 )*xh[0]*dx

R_z = y*2*z*(psi11)*xh[0]*dx + y*(ce)*xh[0]*dx + 3*delta*Gc/8*(-y/eps + 2*eps*ufl.inner(ufl.grad(z),ufl.grad(y)))*xh[0]*dx + ufl.derivative(Wv,z,y)

# Compute Jacobian of R_z
Jac_z = ufl.derivative(R_z, z, dz)


class NonlinearPDEProblem:
    """Nonlinear problem class for a PDE problem using SNES interface."""

    def __init__(self, F, u, bc, J):
        """Initialize nonlinear PDE problem."""
        V = u.function_space
        du = ufl.TrialFunction(V)
        self.L = fem.form(F)
        self.a = fem.form(J)
        self.bc = bc
        self._F, self._J = None, None
        self.u = u

    def F(self, snes, x, F):
        """Assemble residual vector."""

        x.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        x.copy(self.u.x.petsc_vec)
        self.u.x.petsc_vec.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

        with F.localForm() as f_local:
            f_local.set(0.0)
        fem.petsc.assemble_vector(F, self.L)
        fem.petsc.apply_lifting(F, [self.a], bcs = [self.bc], x0 = [x], alpha = -1.0)
        F.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        fem.petsc.set_bc(F, self.bc, x, -1.0)

    def J(self, snes, x, J, P):
        """Assemble Jacobian matrix."""
        x.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        x.copy(self.u.x.petsc_vec)
        self.u.x.petsc_vec.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

        J.zeroEntries()
        fem.petsc.assemble_matrix(J, self.a, bcs = self.bc)
        J.assemble()


# time-stepping parameters
T=1
Totalsteps=1000
startstepsize=1/Totalsteps
stepsize=startstepsize
t=stepsize
step=1
rtol=1e-9
printsteps = 20

def update(solver, dx, x):
    x.axpy(-1, dx)

# Create nonlinear problem
problem_u = NonlinearPDEProblem(R, u, bcs, Jac)

# Create Newton solver and solve


b_u_vec = fem.petsc.create_vector(V)
J_u_mat = fem.petsc.create_matrix(problem_u.a)


solver = PETSc.SNES().create()
solver.setFunction(problem_u.F, b_u_vec)
solver.setJacobian(problem_u.J, J_u_mat)


solver.setTolerances(rtol=1.0e-7, max_it=50)
solver.getKSP().setType("preonly")
solver.getKSP().setTolerances(rtol=1.0e-7)
solver.getKSP().getPC().setType("lu")


# Create nonlinear problem
problem_z = NonlinearPDEProblem(R_z, z, bcs_z, Jac_z)

# Create Newton solver and solve
b_z_vec = fem.petsc.create_vector(Y)
J_z_mat = fem.petsc.create_matrix(problem_z.a)

solver_z = PETSc.SNES().create()
solver_z.setFunction(problem_z.F, b_z_vec)
solver_z.setJacobian(problem_z.J, J_z_mat)

solver_z.setTolerances(rtol=1.0e-7, max_it=50)
solver_z.getKSP().setType("preonly")
solver_z.getKSP().setTolerances(rtol=1.0e-7)
solver_z.getKSP().getPC().setType("lu")

while t-stepsize < T:

    if comm_rank==0:
        print('Step= %d' %step, 't= %f' %t, 'Stepsize= %e' %stepsize)

    bct.g.value[...] = ScalarType(t*maxdisp)
    stag_iter = 1
    rnorm_stag = 1
    while stag_iter<100 and rnorm_stag > 1e-7:
        start_time=time.time()
        ##############################################################
        # PDE for u
        ##############################################################
        u_copy = u.x.petsc_vec.copy()
        u_copy.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        solver.solve(None, u_copy)
        u.x.scatter_forward()
        ##############################################################
        # PDE for z
        ##############################################################
        z_copy = z.x.petsc_vec.copy()
        z_copy.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        solver_z.solve(None, z_copy)
        z.x.scatter_forward()
        ##############################################################

        zmin = domain.comm.allreduce(np.min(z.x.array), op=MPI.MIN)


        if comm_rank==0:
            print(zmin)

        if comm_rank==0:
            print("--- %s seconds ---" % (time.time() - start_time))

        ###############################################################
        #Residual check for stag loop
        ###############################################################
        b_e = fem.petsc.assemble_vector(fem.form(-R))
        fint=b_e.copy()
        fem.petsc.set_bc(b_e, bcs, u.x.petsc_vec, -1.0)


        rnorm_stag=b_e.norm()
        stag_iter+=1


    ########### Post-processing ##############

    u_prev.x.array[:] = u.x.array
    z_prev.x.array[:] = z.x.array

    # Calculate Reaction

    Fx = domain.comm.allreduce(np.sum(fint[y_dofs_top]), op=MPI.SUM)*np.pi*2

    if comm_rank==0:
        print(Fx)
        with open('Indentation_results_axisymmetric.txt', 'a') as rfile:
            rfile.write("%s %s  %s\n" % (str(t), str(zmin), str(Fx)))
    if step % printsteps==0:
        with io.XDMFFile(domain.comm, "paraview/2D_axisymmetric_indentation_" + str(step) + ".xdmf", "w") as file_results:
            file_results.write_mesh(domain)
            file_results.write_function(u, t)
            file_results.write_function(z, t)

    if np.isnan(zmin):
        t1=t
        break

    # time stepping
    step+=1
    t+=stepsize


