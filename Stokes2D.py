# Name : Stokes2D.py
# Author : Emile Soutter
# Use : simple 2D experiment to compute and check the solution of a Stokes experiment.
from __future__ import print_function
from fenics import *
import numpy as np
import matplotlib.pyplot as plt
from mshr import*
from ufl import transpose
import scipy.io as sio
import os.path
from dolfin import*
import pdb
import os
def boundary(x):
    tol = 1E-12
    return abs(x[0]) < tol or abs(x[1]) < tol \
        or abs(x[0] - 1) < tol or abs(x[1] - 1) < tol
def ComputeSolution(N,A_tot,L,bcs,V,Monitor=True):
    #Compute solution, using a direct solver if N is small and iterative one if the matrix is important
    u1=Function(V)
    parms = parameters["krylov_solver"]
    parms["relative_tolerance"]=5.e-6;
    #parms["absolute_tolerance"]=1.e-7;
    parms["maximum_iterations"]=2000;
    parms["monitor_convergence"]=Monitor;
    if(N<1300):
        solve(A_tot==L, u1, bcs,solver_parameters={"linear_solver": "petsc","preconditioner": "icc"})
    else:
        solve(A_tot==L, u1, bcs,solver_parameters={"linear_solver": "tfqmr","preconditioner": "hypre_euclid","krylov_solver": parms})
    return u1
def epsilon(u):
    return 0.5*(nabla_grad(u) + nabla_grad(u).T)
def Compute_StokesSolution2D(N,Amplirho=0.0,gammarho=1.0,rhol=1.0,mu=1.0,AmpliR=1.0,AmpliP=1.0,degree=10,savefiles=0,plotopt=1,order=1):
    deltas= 2.0;
    dimens=2.0;
    facto_dimens=1/dimens;
    print(" | | | Computation of stokes with value N= %.15g" % N);
    print(" | | | Maximum value of density is  %.15g" % rhol);
    print(" | | | Maximum value of the variation of the density is  %.15g" % Amplirho);
    mesh=UnitSquareMesh(N,N);
    #----------------PHYSICAL PARAMETERS-----------------
    # Create mesh and boundaries in Fenics format
    # -------------------------------------------
    h=CellDiameter(mesh)
    n = FacetNormal(mesh)
    zero=Constant(0.)
    #--------------------FUNCTION SPACE---------------------
    # 3 spaces : Vr for the velocity, Qr for the pressure and R for the constraint on the pressure (integral of p over Omega is zero, imposed via Lagrangian multiplier)
    Vr  = VectorElement("CG", mesh.ufl_cell(),order, dim=int(dimens))
    Qr  = FiniteElement("CG", mesh.ufl_cell(),1)
    R=FiniteElement("R",mesh.ufl_cell(),0)
    V   = FunctionSpace(mesh, MixedElement([Vr,Qr,R]))
    u,phi= TrialFunction(V), TestFunction(V)
    u_r,p_r,lambda_r=split(u)
    v_r,q_r,mu_r=split(phi)
    #--------------------BOUNDARY CONDITIONS---------------------
    # Exact analytical expressions for u on the boundary, alternatively can be set to zero in this case.
    u_D1 = Expression(("0.0"), degree=2)
    u_D2 = Expression(("0.0"), degree=2)
    bc1 = DirichletBC(V.sub(0).sub(0), u_D1, boundary)
    bc2 = DirichletBC(V.sub(0).sub(1), u_D2, boundary)
    bc3 = DirichletBC(V.sub(2), zero, boundary)
    bcs=[bc1,bc2,bc3]
    #bcs=[bc3]
    #--------------------DATA OF PROBLEM---------------------
    # Non-trivial expression for the analytical force in the rhs, pressure the solution is p=xy-1/4
    # Expression of the variable density
    rhoexpr=Expression(("(rhol-Amplirho*(0.5*tanh(gammarho*(x[1]-0.5))+0.5))","(rhol-Amplirho*(0.5*tanh(gammarho*(x[1]-0.5))+0.5))","(rhol-Amplirho*(0.5*tanh(gammarho*(x[1]-0.5))+0.5))","0.0"),rhol=rhol,Amplirho=Amplirho,gammarho=gammarho,degree=degree)
    rhof=Function(V)
    rhof.interpolate(rhoexpr)
    rhophantom1,rho,rhophantom2l=rhof.split()
    #Exact expression directly imported from maple. (Huge analytical expression)
    forceexpr = Expression(("-0.2e1 * mu * ((0.24e2 * AmpliR * x[0] * x[0] * pow(x[1], 0.3e1) - 0.36e2 * AmpliR * x[0] * x[0] * x[1] * x[1] - 0.24e2 * AmpliR * x[0] * pow(x[1], 0.3e1) + 0.12e2 * AmpliR * x[0] * x[0] * x[1] + 0.36e2 * AmpliR * x[0] * x[1] * x[1] + 0.4e1 * AmpliR * pow(x[1], 0.3e1) - 0.12e2 * AmpliR * x[0] * x[1] - 0.6e1 * AmpliR * x[1] * x[1] + 0.2e1 * AmpliR * x[1]) / (rhol - Amplirho * (tanh(gammarho * (x[1] - 0.5)) / 0.2e1 + 0.1e1 / 0.2e1)) + (0.12e2 * AmpliR * pow(x[0], 0.4e1) * x[1] - 0.6e1 * AmpliR * pow(x[0], 0.4e1) - 0.24e2 * AmpliR * pow(x[0], 0.3e1) * x[1] + 0.12e2 * AmpliR * pow(x[0], 0.3e1) + 0.12e2 * AmpliR * x[0] * x[0] * x[1] - 0.6e1 * AmpliR * x[0] * x[0]) / (rhol - Amplirho * (tanh(gammarho * (x[1] - 0.5)) / 0.2e1 + 0.1e1 / 0.2e1)) + (0.12e2 * AmpliR * pow(x[0], 0.4e1) * x[1] * x[1] - 0.12e2 * AmpliR * pow(x[0], 0.4e1) * x[1] - 0.24e2 * AmpliR * pow(x[0], 0.3e1) * x[1] * x[1] + 0.2e1 * AmpliR * pow(x[0], 0.4e1) + 0.24e2 * AmpliR * pow(x[0], 0.3e1) * x[1] + 0.12e2 * AmpliR * x[0] * x[0] * x[1] * x[1] - 0.4e1 * AmpliR * pow(x[0], 0.3e1) - 0.12e2 * AmpliR * x[0] * x[0] * x[1] + 0.2e1 * AmpliR * x[0] * x[0]) * Amplirho * gammarho * (0.1e1 - pow(tanh(gammarho * (x[1] - 0.5)), 0.2e1)) * pow(rhol - Amplirho * (tanh(gammarho * (x[1] - 0.5)) / 0.2e1 + 0.1e1 / 0.2e1), -0.2e1) / 0.2e1 + (0.4e1 * AmpliR * pow(x[0], 0.4e1) * pow(x[1], 0.3e1) - 0.6e1 * AmpliR * pow(x[0], 0.4e1) * x[1] * x[1] - 0.8e1 * AmpliR * pow(x[0], 0.3e1) * pow(x[1], 0.3e1) + 0.2e1 * AmpliR * pow(x[0], 0.4e1) * x[1] + 0.12e2 * AmpliR * pow(x[0], 0.3e1) * x[1] * x[1] + 0.4e1 * AmpliR * x[0] * x[0] * pow(x[1], 0.3e1) - 0.4e1 * AmpliR * pow(x[0], 0.3e1) * x[1] - 0.6e1 * AmpliR * x[0] * x[0] * x[1] * x[1] + 0.2e1 * AmpliR * x[0] * x[0] * x[1]) * Amplirho * Amplirho * gammarho * gammarho * pow(0.1e1 - pow(tanh(gammarho * (x[1] - 0.5)), 0.2e1), 0.2e1) * pow(rhol - Amplirho * (tanh(gammarho * (x[1] - 0.5)) / 0.2e1 + 0.1e1 / 0.2e1), -0.3e1) / 0.4e1 - (0.4e1 * AmpliR * pow(x[0], 0.4e1) * pow(x[1], 0.3e1) - 0.6e1 * AmpliR * pow(x[0], 0.4e1) * x[1] * x[1] - 0.8e1 * AmpliR * pow(x[0], 0.3e1) * pow(x[1], 0.3e1) + 0.2e1 * AmpliR * pow(x[0], 0.4e1) * x[1] + 0.12e2 * AmpliR * pow(x[0], 0.3e1) * x[1] * x[1] + 0.4e1 * AmpliR * x[0] * x[0] * pow(x[1], 0.3e1) - 0.4e1 * AmpliR * pow(x[0], 0.3e1) * x[1] - 0.6e1 * AmpliR * x[0] * x[0] * x[1] * x[1] + 0.2e1 * AmpliR * x[0] * x[0] * x[1]) * Amplirho * gammarho * gammarho * tanh(gammarho * (x[1] - 0.5)) * (0.1e1 - pow(tanh(gammarho * (x[1] - 0.5)), 0.2e1)) * pow(rhol - Amplirho * (tanh(gammarho * (x[1] - 0.5)) / 0.2e1 + 0.1e1 / 0.2e1), -0.2e1) / 0.2e1)+AmpliP*x[1]","-0.2e1 * mu * ((-0.24e2 * AmpliR * pow(x[0], 0.3e1) * x[1] * x[1] + 0.24e2 * AmpliR * pow(x[0], 0.3e1) * x[1] + 0.36e2 * AmpliR * x[0] * x[0] * x[1] * x[1] - 0.4e1 * AmpliR * pow(x[0], 0.3e1) - 0.36e2 * AmpliR * x[0] * x[0] * x[1] - 0.12e2 * AmpliR * x[0] * x[1] * x[1] + 0.6e1 * AmpliR * x[0] * x[0] + 0.12e2 * AmpliR * x[0] * x[1] - 0.2e1 * AmpliR * x[0]) / (rhol - Amplirho * (tanh(gammarho * (x[1] - 0.5)) / 0.2e1 + 0.1e1 / 0.2e1)) + (-0.16e2 * AmpliR * pow(x[0], 0.3e1) * pow(x[1], 0.3e1) + 0.24e2 * AmpliR * pow(x[0], 0.3e1) * x[1] * x[1] + 0.24e2 * AmpliR * x[0] * x[0] * pow(x[1], 0.3e1) - 0.8e1 * AmpliR * pow(x[0], 0.3e1) * x[1] - 0.36e2 * AmpliR * x[0] * x[0] * x[1] * x[1] - 0.8e1 * AmpliR * x[0] * pow(x[1], 0.3e1) + 0.12e2 * AmpliR * x[0] * x[0] * x[1] + 0.12e2 * AmpliR * x[0] * x[1] * x[1] - 0.4e1 * AmpliR * x[0] * x[1]) * Amplirho * gammarho * (0.1e1 - pow(tanh(gammarho * (x[1] - 0.5)), 0.2e1)) * pow(rhol - Amplirho * (tanh(gammarho * (x[1] - 0.5)) / 0.2e1 + 0.1e1 / 0.2e1), -0.2e1) / 0.2e1 + (-0.4e1 * AmpliR * pow(x[0], 0.3e1) * pow(x[1], 0.4e1) + 0.8e1 * AmpliR * pow(x[0], 0.3e1) * pow(x[1], 0.3e1) + 0.6e1 * AmpliR * x[0] * x[0] * pow(x[1], 0.4e1) - 0.4e1 * AmpliR * pow(x[0], 0.3e1) * x[1] * x[1] - 0.12e2 * AmpliR * x[0] * x[0] * pow(x[1], 0.3e1) - 0.2e1 * AmpliR * x[0] * pow(x[1], 0.4e1) + 0.6e1 * AmpliR * x[0] * x[0] * x[1] * x[1] + 0.4e1 * AmpliR * x[0] * pow(x[1], 0.3e1) - 0.2e1 * AmpliR * x[0] * x[1] * x[1]) * Amplirho * Amplirho * gammarho * gammarho * pow(0.1e1 - pow(tanh(gammarho * (x[1] - 0.5)), 0.2e1), 0.2e1) * pow(rhol - Amplirho * (tanh(gammarho * (x[1] - 0.5)) / 0.2e1 + 0.1e1 / 0.2e1), -0.3e1) / 0.4e1 - (-0.4e1 * AmpliR * pow(x[0], 0.3e1) * pow(x[1], 0.4e1) + 0.8e1 * AmpliR * pow(x[0], 0.3e1) * pow(x[1], 0.3e1) + 0.6e1 * AmpliR * x[0] * x[0] * pow(x[1], 0.4e1) - 0.4e1 * AmpliR * pow(x[0], 0.3e1) * x[1] * x[1] - 0.12e2 * AmpliR * x[0] * x[0] * pow(x[1], 0.3e1) - 0.2e1 * AmpliR * x[0] * pow(x[1], 0.4e1) + 0.6e1 * AmpliR * x[0] * x[0] * x[1] * x[1] + 0.4e1 * AmpliR * x[0] * pow(x[1], 0.3e1) - 0.2e1 * AmpliR * x[0] * x[1] * x[1]) * Amplirho * gammarho * gammarho * tanh(gammarho * (x[1] - 0.5)) * (0.1e1 - pow(tanh(gammarho * (x[1] - 0.5)), 0.2e1)) * pow(rhol - Amplirho * (tanh(gammarho * (x[1] - 0.5)) / 0.2e1 + 0.1e1 / 0.2e1), -0.2e1) / 0.2e1 + (-0.12e2 * AmpliR * x[0] * pow(x[1], 0.4e1) + 0.24e2 * AmpliR * x[0] * pow(x[1], 0.3e1) + 0.6e1 * AmpliR * pow(x[1], 0.4e1) - 0.12e2 * AmpliR * x[0] * x[1] * x[1] - 0.12e2 * AmpliR * pow(x[1], 0.3e1) + 0.6e1 * AmpliR * x[1] * x[1]) / (rhol - Amplirho * (tanh(gammarho * (x[1] - 0.5)) / 0.2e1 + 0.1e1 / 0.2e1)))+AmpliP*x[0]", "0.0","0.0"),AmpliR=AmpliR,AmpliP=AmpliP,rhol=rhol,Amplirho=Amplirho,gammarho=gammarho,mu=mu, degree=degree)
    forcetot=Function(V)
    forcetot.interpolate(forceexpr)
    force12,forcep,forcel=forcetot.split()
    #--------------------VARIATIONAL PROBLEM---------------------
    #--------------------WEAK FORM--------------------- 
    A_tot = (2*mu*(inner(epsilon(u_r), epsilon(v_r))-(facto_dimens)*inner(div(u_r),div(v_r))) - inner(p_r,div(v_r)) + inner(div(rho*u_r),q_r))*dx+deltas*h*h*inner(grad(p_r),rho*grad(q_r))*dx+inner(p_r,mu_r)*dx+inner(lambda_r,q_r)*dx
    L = inner(force12, v_r)*dx+deltas*h*h*inner(force12,rho*grad(q_r))*dx
    #--------------------Compute SOLUTION---------------------
    u1=ComputeSolution(N,A_tot,L,bcs,V)
    (um, pm,lambdam) = u1.split(True)
    if(savefiles==1):
        # Save solutions in PVD format
        ufile_pvd = File("stokesfenics/velocityN"+str(N)+"rhov"+str(Amplirho)+"gamma"+str(gammarho)+".pvd")
        ufile_pvd << um
        pfile_pvd = File("stokesfenics/pressureN"+str(N)+"rhov"+str(Amplirho)+"gamma"+str(gammarho)+".pvd")
        pfile_pvd << pm
        ffile_pvd = File("stokesfenics/forceN"+str(N)+"rhov"+str(Amplirho)+"gamma"+str(gammarho)+".pvd")
        ffile_pvd << force12
        rhofile_pvd = File("stokesfenics/rhoN"+str(N)+"rhov"+str(Amplirho)+"gamma"+str(gammarho)+".pvd")
        rhofile_pvd << rho
    #Compute error between approximate solution and analytical
    print(" | | | Error computation session:")
    uexact_E=Expression(("(double) (2 * AmpliR * x[1] * pow((double) (x[1] - 1), (double) 2) * x[0] * x[0] * pow((double) (x[0] - 1), (double) 2) + 2 * AmpliR * x[1] * x[1] * (x[1] - 1) * x[0] * x[0] *  pow((double) (x[0] - 1), (double) 2)) / (rhol - Amplirho * (tanh(gammarho * ((double) x[1] - 0.5e0)) / 0.2e1 + 0.1e1 / 0.2e1))","(double) (-2 * AmpliR * x[1] * x[1] * pow((double) (x[1] - 1), (double) 2) * x[0] * pow((double) (x[0] - 1), (double) 2) - 2 * AmpliR * x[1] * x[1] * pow((double) (x[1] - 1), (double) 2) * x[0] * x[0] * (x[0] - 1)) / (rhol - Amplirho * (tanh(gammarho * ((double) x[1] - 0.5e0)) / 0.2e1 + 0.1e1 / 0.2e1))"),AmpliR=AmpliR,rhol=rhol,Amplirho=Amplirho,gammarho=gammarho,degree=degree)
    pexact_E=Expression(("AmpliP*(x[0]*x[1]-0.25)"),AmpliP=AmpliP,degree=degree)
    forceL2=norm(force12,'L2')
    uerrorL2=errornorm(uexact_E,um,'L2', degree_rise=3)
    perrorL2=errornorm(pexact_E,pm,'L2', degree_rise=3)
    uerrorH1=errornorm(uexact_E,um,norm_type = 'H10',degree_rise=3)
    uerrorH1B=errornorm(uexact_E,um,norm_type = 'H1',degree_rise=3)
    print(" | | | L2 norm of the external forces: %.15g" % forceL2)
    print(" | | | Error in L2 norm pressure: %.15g" % perrorL2)
    print(" | | | Error in L2 norm velocity: %.15g" % uerrorL2)
    print(" | | | Error in H1 seminorm velocity: %.15g" % uerrorH1)
    print(" | | | Error in H1 full norm velocity: %.15g" % uerrorH1B)
    #Pointwise saving of the error
    uerror=Function(V)
    uexact_tot=Function(V)
    uexactex_E=Expression(("((double) (2 * AmpliR * x[1] * pow((double) (x[1] - 1), (double) 2) * x[0] * x[0] * pow((double) (x[0] - 1), (double) 2) + 2 * AmpliR * x[1] * x[1] * (x[1] - 1) * x[0] * x[0] *  pow((double) (x[0] - 1), (double) 2)) / (rhol - Amplirho * (tanh(gammarho * ((double) x[1] - 0.5e0)) / 0.2e1 + 0.1e1 / 0.2e1)))-um1","((double) (-2 * AmpliR * x[1] * x[1] * pow((double) (x[1] - 1), (double) 2) * x[0] * pow((double) (x[0] - 1), (double) 2) - 2 * AmpliR * x[1] * x[1] * pow((double) (x[1] - 1), (double) 2) * x[0] * x[0] * (x[0] - 1)) / (rhol - Amplirho * (tanh(gammarho * ((double) x[1] - 0.5e0)) / 0.2e1 + 0.1e1 / 0.2e1)))-um2","AmpliP*(x[0]*x[1]-0.25)-pm","0.0"),AmpliR=AmpliR,rhol=rhol,Amplirho=Amplirho,gammarho=gammarho,um1=um.sub(0),um2=um.sub(1),AmpliP=AmpliP,pm=pm,degree=degree)
    uexactot_E=Expression(("((double) (2 * AmpliR * x[1] * pow((double) (x[1] - 1), (double) 2) * x[0] * x[0] * pow((double) (x[0] - 1), (double) 2) + 2 * AmpliR * x[1] * x[1] * (x[1] - 1) * x[0] * x[0] *  pow((double) (x[0] - 1), (double) 2)) / (rhol - Amplirho * (tanh(gammarho * ((double) x[1] - 0.5e0)) / 0.2e1 + 0.1e1 / 0.2e1)))","((double) (-2 * AmpliR * x[1] * x[1] * pow((double) (x[1] - 1), (double) 2) * x[0] * pow((double) (x[0] - 1), (double) 2) - 2 * AmpliR * x[1] * x[1] * pow((double) (x[1] - 1), (double) 2) * x[0] * x[0] * (x[0] - 1)) / (rhol - Amplirho * (tanh(gammarho * ((double) x[1] - 0.5e0)) / 0.2e1 + 0.1e1 / 0.2e1)))","AmpliP*(x[0]*x[1]-0.25)","0.0"),AmpliR=AmpliR,rhol=rhol,Amplirho=Amplirho,gammarho=gammarho,AmpliP=AmpliP,degree=degree)
    uexact_tot.interpolate(uexactot_E)
    uerror.interpolate(uexactex_E)
    umerr,perr,lerr=uerror.split()
    umex,pex,lex=uexact_tot.split()
    if(savefiles==1):
        lfile_pvd = File("stokesfenics/velocityerror"+str(N)+"rhov"+str(Amplirho)+"gamma"+str(gammarho)+".pvd")
        lfile_pvd << umerr
        perfile_pvd = File("stokesfenics/pressureerror"+str(N)+"rhov"+str(Amplirho)+"gamma"+str(gammarho)+".pvd")
        perfile_pvd << perr
        uexfile_pvd = File("stokesfenics/velocityexact"+str(N)+"rhov"+str(Amplirho)+"gamma"+str(gammarho)+".pvd")
        uexfile_pvd<<umex
        uexfile_pvd = File("stokesfenics/pressureexact"+str(N)+"rhov"+str(Amplirho)+"gamma"+str(gammarho)+".pvd")
        uexfile_pvd<<pex
    # Plot solution
    if(plotopt==1):
        #Pression
        plt.figure()
        fig1=plot(pm,scale = 2.0, title = "Plot of the pressure" )
        plt.colorbar(fig1)
        plt.savefig("stokesfenics/Images/pressureN"+str(N)+"rhov"+str(Amplirho)+"gamma"+str(gammarho)+".png")
        #Velocity
        plt.figure()
        fig2=plot(um,scale = 2.0, title = "Plot of the velocity" )
        plt.colorbar(fig2)
        plt.savefig("stokesfenics/Images/velocityN"+str(N)+"rhov"+str(Amplirho)+"gamma"+str(gammarho)+".png")
        #plt.show()
    return uerrorL2,perrorL2,uerrorH1

#Main algorithm : parameters and boucle for convergence
Amplirho=0.9;
gamma=1;
Maxconv=15; # number of meshes
MinvalueN=10; # minimum N
uL2=np.zeros(Maxconv);
uH1=np.zeros(Maxconv);
pL2=np.zeros(Maxconv);
#Make the required folders to save data if not existing
if not os.path.exists('stokesfenics'):
    os.makedirs('stokesfenics')
    print("Folder stokesfenics juste created.")
if not os.path.exists('stokesfenics/Images'):
    os.makedirs('stokesfenics/Images')
    print("Folder stokesfenics/Images juste created.")
Nexp=np.arange(Maxconv);
Nvector=np.array(MinvalueN*(pow(2,Nexp*0.5)),dtype=int)
for kk in range (0,Maxconv):
    saveparam=0;
    if(kk==4):
        saveparam=1;
    if(kk==5):
        saveparam=1;
    if(kk==Maxconv-1):
        saveparam=1;
    uL2[kk],pL2[kk],uH1[kk]=Compute_StokesSolution2D(Nvector[kk],order=1,Amplirho=Amplirho,savefiles=saveparam,gammarho=gamma);
    #pdb.set_trace()
pdb.set_trace()
