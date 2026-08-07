from mpi4py import MPI
from dolfinx import mesh, fem, io, plot, nls, log, geometry, la
from dolfinx import cpp as _cpp
from dolfinx import default_real_type
import basix
import dolfinx.fem.petsc
import ufl
import numpy as np
from petsc4py.PETSc import ScalarType
from petsc4py import PETSc
import time
import os
import gmsh
log.set_log_level(log.LogLevel.WARNING)
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


# Material properties
E, nu = ScalarType(70000), ScalarType(0.22)	                                    #Young's modulus and Poisson's ratio
mu, lmbda, kappa = E/(2*(1 + nu)), E*nu/((1 + nu)*(1 - 2*nu)), E/(3*(1 - 2*nu))
Gc= ScalarType(0.010)	                                                          #Critical energy release rate
sts, scs= ScalarType(40), ScalarType(1000)	                                      #Tensile strength and compressive strength
shs = (2/3)*sts*scs/(scs-sts)
Wts = sts**2/(2*E)
Whs = shs**2/(2*kappa)



#Irwin characteristic length
lch=3*Gc*E/8/(sts**2)
#The regularization length
eps = 0.16
h = 0.05



delta = (1+3*h/(8*eps))**(-2) * ((sts + (1+2*np.sqrt(3))*shs)/((8+3*np.sqrt(3))*shs)) * 3*Gc/(16*Wts*eps) + (1+3*h/(8*eps))**(-1) * (2/5)
comm = MPI.COMM_WORLD
comm_rank = MPI.COMM_WORLD.rank
log.set_log_level(log.LogLevel.ERROR)

#Geometry

# Parameters for the outer rectangle


L1 = 1.5
L2 = 50
L = L1 + L2

H = 10
B = 2.5/2

x_cyl = L1
y_cyl = H/3
z_cyl = B
r_cyl = B / 4

ac = 25
ac += L1




gmsh.initialize()
    
gmsh.model.add("DCB")


# Create outer box
block = gmsh.model.occ.addRectangle(0, 0, 0, L, H)

# Create inner cylinder (same axis, smaller radius)
inner_cyl = gmsh.model.occ.addDisk(x_cyl, y_cyl, 0, r_cyl, r_cyl)

# Cut inner cylinder from outer cylinder to form a tube
dcb, _ = gmsh.model.occ.cut([(2, block)], [(2, inner_cyl)])

# Synchronize to reflect the changes in the model
gmsh.model.occ.synchronize()

# Add physical group for the volume (the tube itself)
dcb_volumes = [entity[1] for entity in dcb]
dcb_group = gmsh.model.addPhysicalGroup(2, dcb_volumes)
gmsh.model.setPhysicalName(2, dcb_group, "dcbVolume")


# Define mesh size fields
field_id = gmsh.model.mesh.field.add("Box")
gmsh.model.mesh.field.setNumber(field_id, "VIn", h)
gmsh.model.mesh.field.setNumber(field_id, "VOut", 4 * h)
gmsh.model.mesh.field.setNumber(field_id, "XMin", ac - 2*eps)
gmsh.model.mesh.field.setNumber(field_id, "XMax", L)
gmsh.model.mesh.field.setNumber(field_id, "YMin",0)
gmsh.model.mesh.field.setNumber(field_id, "YMax", H/3)



field2_id = gmsh.model.mesh.field.add("Box")
gmsh.model.mesh.field.setNumber(field2_id, "VIn", 2 * h)
gmsh.model.mesh.field.setNumber(field2_id, "VOut", 4 * h)
gmsh.model.mesh.field.setNumber(field2_id, "XMin", 0)
gmsh.model.mesh.field.setNumber(field2_id, "XMax", L)
gmsh.model.mesh.field.setNumber(field2_id, "YMin", 0)
gmsh.model.mesh.field.setNumber(field2_id, "YMax", H/2)


field3_id = gmsh.model.mesh.field.add("Box")
gmsh.model.mesh.field.setNumber(field3_id, "VIn", 2* h)
gmsh.model.mesh.field.setNumber(field3_id, "VOut", 4 * h)
gmsh.model.mesh.field.setNumber(field3_id, "XMin", x_cyl - 1.1*r_cyl)
gmsh.model.mesh.field.setNumber(field3_id, "XMax", x_cyl + 1.1*r_cyl)
gmsh.model.mesh.field.setNumber(field3_id, "YMin", y_cyl - 1.1*r_cyl)
gmsh.model.mesh.field.setNumber(field3_id, "YMax", y_cyl + 1.1*r_cyl)




# Combine the fields
min_field_id = gmsh.model.mesh.field.add("Min")
gmsh.model.mesh.field.setNumbers(min_field_id, "FieldsList", [field_id, field2_id, field3_id])
gmsh.model.mesh.field.setAsBackgroundMesh(min_field_id)


# Generate and optimize the mesh
gmsh.model.mesh.generate(2)
gmsh.model.mesh.optimize("Netgen")


model = MPI.COMM_WORLD.bcast(gmsh.model, root=0)
partitioner = dolfinx.cpp.mesh.create_cell_partitioner(mesh.GhostMode.shared_facet, None)
mesh_data = io.gmsh.model_to_mesh(model, MPI.COMM_WORLD, 0, gdim=2, partitioner=partitioner)


gmsh.finalize()
domain = mesh_data[0]


with dolfinx.io.XDMFFile(domain.comm, "refined_mesh_DCB.xdmf", "w") as xdmf:
    xdmf.write_mesh(domain)



# Defining the function spaces
V = fem.functionspace(domain, ("CG", 1, (domain.geometry.dim,)))                  #Function space for u
Y = fem.functionspace(domain, ("CG", 1))                                          #Function space for z



def bottom(x):
    return (x[1]<1e-4) & (x[0]>ac - 1e-4)

def cracktip(x):
    return np.logical_and.reduce((
        x[1] < 1e-4,
        x[0] > ac - eps,
        x[0] < ac + h
    ))



def ring(x):
    return np.logical_and(
        np.isclose((x[0] - x_cyl)**2 + (x[1] - y_cyl)**2, r_cyl**2),
        x[1] > y_cyl
    )


def corner(x):
    return (x[1]<1e-4) & (np.isclose(x[0],L))

def outer(x):
    return (x[0] < x_cyl + r_cyl + 1e-4)

fdim = domain.topology.dim -1

bottom_facets = mesh.locate_entities_boundary(domain, fdim, bottom)
cracktip_facets = mesh.locate_entities_boundary(domain, fdim, cracktip)

corner_facets = mesh.locate_entities_boundary(domain, 0, corner)


ring_facets = mesh.locate_entities_boundary(domain, fdim, ring)
outer_facets = mesh.locate_entities(domain, fdim, outer)



dofs_bottom0 = fem.locate_dofs_topological(V.sub(0), fdim, bottom_facets)
dofs_bottom1 = fem.locate_dofs_topological(V.sub(1), fdim, bottom_facets)

dofs_ring0 = fem.locate_dofs_topological(V.sub(0), fdim, ring_facets)
dofs_ring1 = fem.locate_dofs_topological(V.sub(1), fdim, ring_facets)

dofs_corner0 = fem.locate_dofs_topological(V.sub(0), 0, corner_facets)
dofs_corner1 = fem.locate_dofs_topological(V.sub(1), 0, corner_facets)

dofs_outer = fem.locate_dofs_topological(Y, fdim, outer_facets)
dofs_cracktip = fem.locate_dofs_topological(Y, fdim, cracktip_facets)

bcb = fem.dirichletbc(ScalarType(0), dofs_bottom1, V.sub(1))
bcr = fem.dirichletbc(ScalarType(0), dofs_ring0, V.sub(0))
bct = fem.dirichletbc(ScalarType(0), dofs_ring1, V.sub(1))

bccorner0 = fem.dirichletbc(ScalarType(0), dofs_corner0, V.sub(0))



bcs = [bcb, bct, bccorner0]


bct_z = fem.dirichletbc(ScalarType(1), dofs_outer, Y)
bct_z2 = fem.dirichletbc(ScalarType(0), dofs_cracktip, Y)
bcs_z = [bct_z, bct_z2]



marked_facets = np.hstack([ring_facets, bottom_facets])
marked_values = np.hstack([np.full_like(ring_facets, 1),
                           np.full_like(bottom_facets, 2)])
sorted_facets = np.argsort(marked_facets)
facet_tag = mesh.meshtags(domain, domain.topology.dim -1,
                          marked_facets[sorted_facets],
                          marked_values[sorted_facets])


metadata = {"quadrature_degree": 2}
ds = ufl.Measure('ds', domain=domain,
                 subdomain_data=facet_tag, metadata=metadata)
dS = ufl.Measure("dS", domain=domain, metadata=metadata)
dx = ufl.Measure("dx", domain=domain, metadata=metadata)

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



un = fem.Function(V)
un.x.array[:] = u.x.array
zn = fem.Function(Y)
zn.x.array[:] = z.x.array


error_u = fem.Function(V)
error_u.x.array[:] = 0
error_z = fem.Function(Y)
error_z.x.array[:] = 0


Vplot = fem.functionspace(domain, ("Lagrange", 1, (domain.geometry.dim,)))        #Function space for plotting u
uplot = fem.Function(Vplot, name="displacement")


def norm_L2(comm, v):
    """Compute the L2(O)-norm of v"""
    return np.sqrt(comm.allreduce(fem.assemble_scalar(fem.form(ufl.inner(v, v) * dx)), op=MPI.SUM))


def local_project(v, V):
    """[summary]
        Helper function to do a interpolation
    Args:
        v ([dolfin.Funcion]): [function to be projected]
        V ([dolfin.Function]): [target `dolfin.FunctionSpace` to be projected on]

    Returns:
        [dolfin.Function]: [target function after projection]
    """
    expr = fem.Expression(v, V.element.interpolation_points, comm)
    u = fem.Function(V)
    u.interpolate(expr)
    return u

def adjust_array_shape(input_array):
    if input_array.shape == (2,):                                                 # Check if the shape is (2,)
        adjusted_array = np.append(input_array, 0.0)                              # Append 0.0 to the array
        return adjusted_array
    else:
        return input_array

bb_tree = geometry.bb_tree(domain, domain.topology.dim)


def evaluate_function(u, x):
    """Evaluates a function at a point `x` in parallel using MPI

    Args:
        u (dolfin.Function): Function to be evaluated
        x (Union(tuple, list, numpy.ndarray)): Point at which to evaluate function `u`

    Returns:
        numpy.ndarray: Function evaluated at point `x`
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

    u_value_local = None

    cells = []
    # Find cells whose bounding-box collide with the points
    cell_candidates = geometry.compute_collisions_points(bb_tree, points)
    # Choose one of the cells that contains the point
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points)

    if len(colliding_cells.links(0)) > 0:
        u_value_local = u.eval(points, colliding_cells.links(0)[0])

    # Gather results from all processes
    u_value_all = comm.gather(u_value_local, root=0)

    # Concatenate results on root process
    if rank == 0:
        # Filter out None values before concatenation
        u_value_all = [arr for arr in u_value_all if arr is not None]
        # Concatenate arrays only if there are non-None values and if there's at least one valid value
        if u_value_all:
            u_value = np.concatenate(u_value_all[:1])  # Take the first valid value
        else:
            u_value = None
    else:
        u_value = None

    # Broadcast the final result to all processes
    u_value = comm.bcast(u_value, root=0)

    return u_value


W0 = fem.functionspace(domain, ("P", 1))



# Stored energy, strain and stress functions in linear isotropic elasticity (plane stress)

def energy(v):
	  return mu*(ufl.inner(ufl.sym(ufl.grad(v)),ufl.sym(ufl.grad(v))) + ((nu/(1-nu))**2)*(ufl.tr(ufl.sym(ufl.grad(v))))**2 )+ 0.5*(lmbda)*(ufl.tr(ufl.sym(ufl.grad(v)))*(1-2*nu)/(1-nu))**2

def epsilon(v):
	return ufl.sym(ufl.grad(v))

def sigma(v):
	return 2.0*mu*ufl.sym(ufl.grad(v)) + (lmbda)*ufl.tr(ufl.sym(ufl.grad(v)))*(1-2*nu)/(1-nu)*ufl.Identity(len(v))

def sigmavm(sig,v):
	return ufl.sqrt(1/2*(ufl.inner(sig-1/3*ufl.tr(sig)*ufl.Identity(len(v)), sig-1/3*ufl.tr(sig)*ufl.Identity(len(v))) + (1/9)*ufl.tr(sig)**2 ))

eta = 0.0
# Stored energy density
psi1 = (z**2+eta)*(energy(u))
psi11 = energy(u)
# Total potential energy
Pi = psi1*dx
# Compute first variation of Pi (directional derivative about u in the direction of v)
R = ufl.derivative(Pi, u, v)
# Compute Jacobian of R
Jac = ufl.derivative(R, u, du)


I1 = (z**2)*ufl.tr(sigma(u))
SQJ2 = (z**2)*sigmavm(sigma(u),u)

alpha1 = (delta*Gc)/(shs*8*eps) - (2*Whs)/(3*shs)
alpha2 = (3**0.5*(3*shs - sts)*delta*Gc)/(shs*sts*8*eps) + (2*Whs)/(3**0.5*shs) - (2*3**0.5*Wts)/(sts)

ce= alpha2*SQJ2 + alpha1*I1 - z*(1-ufl.sqrt(I1**2)/I1)*psi11

#Balance of configurational forces PDE
pen=1000*(3*Gc/8/eps)*ufl.conditional(ufl.lt(delta,1),1, delta)
Wv=pen/2*((abs(z)-z)**2 + (abs(1-z) - (1-z))**2 )*dx

R_z = y*2*z*(psi11)*dx + y*(ce)*dx + 3*delta*Gc/8*(-y/eps + 2*eps*ufl.inner(ufl.grad(z),ufl.grad(y)))*dx + ufl.derivative(Wv,z,y)

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
ldot = 5*10**(-1)
maxdisp = 0.05

# time-stepping parameters
T = maxdisp / (ldot)


Totalsteps = 40
startstepsize=T/Totalsteps
stepsize=startstepsize
t=stepsize
step=1
rnorm_stag0 = 1
rnorm_stag = 1
printsteps = 100
printsteps2 = 1




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


vtk_v = io.VTKFile(domain.comm, "Files_DCB/Paraview/2D_DCB.pvd", "a")

while t-stepsize < T:

    if comm_rank==0:
        print('Step= %d' %step, 't= %f' %t, 'Stepsize= %e' %stepsize)


    bct.g.value[...] = ScalarType(t/T*maxdisp)

    stag_iter = 1
    stag_iter = 1
    rnorm_stag = 1
    while stag_iter<100 and rnorm_stag/rnorm_stag0 > 1e-7:
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


        u_prev.x.array[:] = u.x.array
        z_prev.x.array[:] = z.x.array
        stag_iter+=1


    ########### Post-processing ##############

    un.x.array[:] = u.x.array
    zn.x.array[:] = z.x.array

    # Calculate Reaction

    Fx = -domain.comm.allreduce(np.sum(fint[dofs_ring1]), op=MPI.SUM)
    
    z_x = evaluate_function(z, (ac + eps, 0))[0]



    if rank==0:
        print(Fx)
        print(z_x)
        with open('Files_DCB/Elastic_phasefield_DCB2D.txt', 'a') as rfile:
            rfile.write("%s %s %s %s %s\n" % (str(t), str(t/T*maxdisp), str(zmin), str(z_x), str(Fx)))




    if step % printsteps2==0:
        uplot.x.array[:] = (local_project(u, Vplot)).x.array
        vtk_v.write_function([uplot, z], t)
        

    # time stepping
    step+=1
    t+=stepsize
vtk_v.close()