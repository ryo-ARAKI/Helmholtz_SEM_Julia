#=
Helmholtz equation in 1D
Solve
    d^2u/dx^2 - λ^2 u = f in x = [a,b]
for
    λ^2 = 0
    f = cos(πx+π/4)
    x = [-1,1]
    u[-1] = 0
    u[1]  = 0
=#

"""
Module for parameters and variables
"""
module  mod_param_var
using OffsetArrays
    struct Parameters
        N_ele::Int64  # Number of elements
        N_x::Int64  # Number of calculation point
        ν::Float64  #
        Δt::Float64  # Time step
        λ::Float64  # Helmholtz constant
        L::Float64  # Range of solution domain
        Li::Float64  # Range of one element (EQUAL FOR ALL ELEMENTS)
    end

    mutable struct Variables
        x::OffsetArray{Float64}  # Global coordinate
        C_local::OffsetArray{Float64, 3}  # Local coefficient matrix for LHS
        B_local::OffsetArray{Float64, 3}  # Local coefficient matrix for RHS
        f_local::OffsetArray{Float64, 2}  # Local RHS vector
        C_global::OffsetArray{Float64, 2}  # Global coefficient matrix for LHS
        B_global::OffsetArray{Float64, 2}  # Global coefficient matrix for RHS
        f_global::OffsetArray{Float64}  # Global RHS vector
        RHS::OffsetArray{Float64}  # Global RHS vector
        u::OffsetArray{Float64}  # Global Solution vector
    end

    mutable struct StatisticalValues
        φ::Float64  # Average of θ: Angle of particles
    end
end  # module mod_param_var


"""
Module for 1-D Helmholtz equation
"""
module mod_1d_helmholtz
using OffsetArrays
using LinearAlgebra
    """
    Calculate local coordinate
    Eq. (4a)
    """
    function local_x(param,j)
        return cos(π*j/param.N_x)
    end

    """
    Calculate global coordinate
    Eq. (4b)
    """
    function global_x(param,i,j)
        a = -param.L/2.0 + (i-1.0) * (param.L/param.N_ele)
        return param.Li/2.0 * (local_x(param,j) + 1.0) + a
    end

    """
    Set initial condition
    """
    function set_initial_condition(param,var)
        var.x = zeros(0:param.N_ele*param.N_x)
        var.C_local = zeros(1:param.N_ele, 0:param.N_x, 0:param.N_x)
        var.B_local = zeros(1:param.N_ele, 0:param.N_x, 0:param.N_x)
        var.f_local = zeros(1:param.N_ele, 0:param.N_x)
        var.C_global = zeros(0:param.N_ele*param.N_x, 0:param.N_ele*param.N_x)
        var.B_global = zeros(0:param.N_ele*param.N_x, 0:param.N_ele*param.N_x)
        var.f_global = zeros(0:param.N_ele*param.N_x)
        var.RHS = zeros(0:param.N_ele*param.N_x)
        var.u = zeros(0:param.N_ele*param.N_x)
    end

    """
    Eq. (6b)
    """
    function coef_c(param,k)
        if k==0
            r = 2.0
        elseif k==param.N_x
            r = 2.0
        else
            r = 1.0
        end
        return r
    end

    """
    Eq. (16b)
    """
    function J_k(k)
        r = 0.0
        if k>=1
            for i=1:k
                r += 1.0/(2.0 * i - 1.0)
            end
            r = -4.0 * r
        elseif k==0
            r = 0.0
        end
    return r
    end

    """
    Eq. (16a)
    """
    function a_nm(n, m)
        dif_m_n = abs(n-m)
        sum_m_n = n+m
        if mod(dif_m_n,2)==0
            r = J_k(dif_m_n/2) - J_k(sum_m_n/2)
            r = n * m / 2.0 * r
        elseif mod(dif_m_n,2)==1
            r = 0.0
        end
        return r
    end

    """
    Eq. (17)
    """
    function b_nm(n, m)
        dif_m_n = abs(n-m)
        sum_m_n = n+m
        if mod(dif_m_n,2)==0
            r = 1.0/(1.0 - sum_m_n^2) + 1.0/(1.0 - dif_m_n^2)
        elseif mod(dif_m_n,2)==1
            r = 0.0
        end
        return r
    end

    """
    Calculate Chebyshev polynominal
    """
    function Tn(n,r)
        return cos(n*acos(r))
    end

    """
    Eq. (15a)
    """
    function tilde_A_ik(param,var,j,k)
        r = 0.0
        for n=0:param.N_x
            for m=0:param.N_x
                r += 1.0/(coef_c(param,n) * coef_c(param,m)) * Tn(n, local_x(param,j)) * Tn(m, local_x(param,k)) * a_nm(n,m)
            end
        end
        r = 4.0 / (coef_c(param,j) * coef_c(param,k)) * r
        return r
    end

    """
    Eq. (15b)
    """
    function tilde_B_ik(param,var,j,k)
        r = 0.0
        for n=0:param.N_x
            for m=0:param.N_x
                r += 1.0/(coef_c(param,n) * coef_c(param,m)) * Tn(n, local_x(param,j)) * Tn(m, local_x(param,k)) * b_nm(n,m)
            end
        end
        r = 4.0 / (coef_c(param,j) * coef_c(param,k)) * r
        return r
    end

    """
    Eq. (15a)
    """
    function A_jk(param,var,j,k)
        return -2.0/(param.Li * param.N_x^2) * tilde_A_ik(param,var,j,k)
    end

    """
    Eq. (15b)
    """
    function B_jk(param,var,j,k)
        return param.Li/(2.0 * param.N_x^2) * tilde_B_ik(param,var,j,k)
    end

    """
    Eq. (14b)
    """
    function C_jk(param,var,j,k)
        return A_jk(param,var,j,k) - param.λ^2 * B_jk(param,var,j,k)
    end

    """
    Eq. (14a)
    """
    function set_B_local(param,var)
        for e=1:param.N_ele
            for j=0:param.N_x
                for k=0:param.N_x
                    var.B_local[e,j,k] = B_jk(param,var,j,k)
                end
            end
        end
    end

    """
    Eq. (14a)
    """
    function set_C_local(param,var)
        for e=1:param.N_ele
            for j=0:param.N_x
                for k=0:param.N_x
                    var.C_local[e,j,k] = C_jk(param,var,j,k)
                end
            end
        end
    end

    """
    Eq. (18a), (18b)
    """
    function set_B_global(param,var)
        for e=1:param.N_ele
            var.B_global[(e-1)*param.N_x:e*param.N_x, (e-1)*param.N_x:e*param.N_x] += var.B_local[e, 0:param.N_x, 0:param.N_x]
        end
        # Enforce Dirichlet boundary condition
        var.B_global[0,0:param.N_ele*param.N_x] = zeros(0:param.N_ele*param.N_x)
    end

    """
    Eq. (18a), (18b)
    """
    function set_C_global(param,var)
        for e=1:param.N_ele
            var.C_global[(e-1)*param.N_x:e*param.N_x, (e-1)*param.N_x:e*param.N_x] += var.C_local[e, 0:param.N_x, 0:param.N_x]
        end
        # Enforce Dirichlet boundary condition
        var.C_global[0,0:param.N_ele*param.N_x] = zeros(0:param.N_ele*param.N_x) # Inlet boundary
        var.C_global[param.N_ele*param.N_x,0:param.N_ele*param.N_x] = zeros(0:param.N_ele*param.N_x)  # Outlet boundary
    end

    """
    Eq. (18a), (18b)
    """
    function set_f_local(param,var)
        for e=1:param.N_ele
            for i=0:param.N_x
                var.f_local[e,param.N_x-i] = cos(π*global_x(param,e,i) + π/4.0)
            end
        end
    end

    """
    Eq. (18a), (18b)
    """
    function set_f_global(param,var)
        for e=1:param.N_ele
                var.f_global[(e-1)*param.N_x:e*param.N_x] = var.f_local[e,0:param.N_x]
        end
    end

    """
    Eq. (18a)
    """
    function set_RHS(param,var)
        var.RHS[0:param.N_ele*param.N_x] = var.B_global[0:param.N_ele*param.N_x, 0:param.N_ele*param.N_x] * var.f_global[0:param.N_ele*param.N_x]
    end

    """
    Solve linear equation by Gauss-Seidel method
    """
    function solve_LineraEquation(param,var)
        # Set initial solution
        n = param.N_ele*param.N_x
        y = zeros(0:n)
        r = zeros(0:n)
        rd = zeros(0:n)
        for i=0:n
            rd[i] = 1.0/var.C_global[i,i]
            if var.C_global[i,i] == 0.0
                rd[i] = 0.0
            end
        end

        # Calculate i-th solution
        for itr=1:10000
            for i=1:n
                s_bef = dot(var.C_global[i,1:i-1], y[1:i-1])
                s_aft = dot(var.C_global[i,i+1:n], y[i+1:n])
                s = s_bef + s_aft
                y[i] = rd[i] * (var.RHS[i]-s)
            end
            # Calculate residual
            r[0:n] = var.RHS[0:n] - var.C_global[0:n, 0:n] * y[0:n]
            er = dot(r[0:n],r[0:n])/(n+1)
            # Cobvergence evaluation
            if er < 10.0^(-10)
                println("Converged, itr= ", itr, " residual= ", er)
                break
            end
            # Output
            # if mod(itr,100) == 0
            #     println("itr= ",itr, " residual= ",er)
            # end
        end
        return y
    end

    """
    Calculate and store global coordinate
    """
    function set_global_x(param,var)
        for e=1:param.N_ele
            for i=0:param.N_x
                var.x[(e-1)*param.N_x+i] = global_x(param,e,param.N_x-i)
            end
        end
    end

end  # module mod_1d_wave


"""
Module for dat, image and movie generation
"""
module mod_output
    using OffsetArrays
    using Plots
    gr(
        # aspect_ratio = 1,
        # legend = false,
        # xaxis=nothing,
        # yaxis=nothing
    )
    """
    Output snapshot image of particle distribution and direction
    """
    function out_snapimg(param,var)
        sol_a = OffsetArray{Float64}(undef, 0:param.N_ele*param.N_x)
        for i=0:param.N_ele*param.N_x
            sol_a[i] = (sin(π*var.x[i])-cos(π*var.x[i])-1.0)/(sqrt(2)*π^2)
        end
        plot(
            var.x[0:param.N_ele*param.N_x],
            [
            var.u[0:param.N_ele*param.N_x],
            sol_a[0:param.N_ele*param.N_x],
            ],
            linewidth = 3
            )
        str_ele = lpad(string(param.N_ele), 3, "0")
        str_x = lpad(string(param.N_x), 3, "0")
        png("result/helmholtz_$(str_ele)_$(str_x).png")
    end
end  # module mod_output


## Declare modules
using OffsetArrays
using ProgressMeter
using .mod_param_var  # Define parameters and variables
import .mod_1d_helmholtz:  # Definde 1-D wave equation
global_x,
set_initial_condition,
set_B_local,
set_C_local,
set_B_global,
set_C_global,
set_f_local,
set_f_global,
set_RHS,
solve_LineraEquation,
set_global_x
import .mod_output:  # Define functions for output data
out_snapimg


## Set parameter
N_ele = 2
N_x = 10
ν = 0.2
Δt = 0.005
λ = 0.0 #sqrt(2.0/(ν*Δt))
L = 2.0
Li = L/N_ele
param_ = mod_param_var.Parameters(N_ele,N_x,ν,Δt,λ,L,Li)

## Set variables
x = OffsetArray{Float64}(undef, 0:param_.N_ele*param_.N_x)
C_local = OffsetArray{Float64}(undef, 1:param_.N_ele, 0:param_.N_x, 0:param_.N_x)
B_local = OffsetArray{Float64}(undef, 1:param_.N_ele, 0:param_.N_x, 0:param_.N_x)
f_local = OffsetArray{Float64}(undef, 1:param_.N_ele, 0:param_.N_x)
C_global = OffsetArray{Float64}(undef, 0:param_.N_ele*param_.N_x, 0:param_.N_ele*param_.N_x)
B_global = OffsetArray{Float64}(undef, 0:param_.N_ele*param_.N_x, 0:param_.N_ele*param_.N_x)
f_global = OffsetArray{Float64}(undef, 0:param_.N_ele*param_.N_x)
RHS = OffsetArray{Float64}(undef, 0:param_.N_ele*param_.N_x)
u = OffsetArray{Float64}(undef, 0:param_.N_ele*param_.N_x)
var_ = mod_param_var.Variables(x,C_local,B_local,f_local,C_global,B_global,f_global,RHS,u)


## Main
set_initial_condition(param_,var_)
set_global_x(param_,var_)

set_B_local(param_,var_)
set_C_local(param_,var_)
set_B_global(param_,var_)
set_C_global(param_,var_)
set_f_local(param_,var_)
set_f_global(param_,var_)
set_RHS(param_,var_)
var_.u = solve_LineraEquation(param_,var_)
# println("C= ")
# for i=0:param_.N_ele*param_.N_x
#     println(var_.C_global[i,0:param_.N_ele*param_.N_x])
# end
# println("RHS=  ",var_.RHS)
# println("u=  ",var_.u)
out_snapimg(param_,var_)
