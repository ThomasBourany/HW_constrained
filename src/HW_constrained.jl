

# constrained maximization exercises

## portfolio choice problem

module HW_constrained

	using JuMP, NLopt, DataFrames

	export data, table_NLopt, table_JuMP

	function data(a=0.5)
	
pi = 1/16.*ones(16)	
z1 = ones(16)
z2 = [0.72,0.92,1.12,1.32]
z3 = [0.86,0.96,1.06,1.16]
z = hcat(z1,kron(z2, ones(4)), kron(ones(4), z3))
p = ones(3)
e = [2.0,0.0,0.0]
truth = DataFrame(a = [0.5,1.0,5.0], c = [1.00801,1.00401,1.0008], omega1 = [-1.41237,-0.206197,0.758762], omega2 =[0.801458,0.400729,0.0801456], omega3 = [1.60291,0.801462,0.160291], fval = [-1.20821,-0.732819,-0.013422])

return Dict("pi"=>pi, "a"=>a, "payoff"=>z, "price" => p, "endow" => e, "truth" => truth)

	end
#dat = data()
	function max_JuMP(a=0.5)

d = data(a)
m = Model()

@defVar(m, c >= 0.0)
@defVar(m, w[1:3])
@setNLObjective(m,:Max, - exp(-a*c) - sum{d["pi"][s]*(exp(-a * sum{w[i]*d["payoff"][s,i],i=1:3})),s=1:16})
## - exp(-a*c) - (d["pi"]'*exp(-a.*(d["payoff"]*w)))[1]
# for once I do a function with vectors, it does not work here...
@addNLConstraint(m, c + sum{d["price"][i]*(w[i] - d["endow"][i]),i=1:3} == 0.0)
## similarly, at first I would have done: c+ (d["price"]'(w[i] - d["endow"]))[1]
solve(m)

return Dict("a" => a, "c" => getValue(c), "w1" => getValue(w[1]), "w2" => getValue(w[2]), "w3" => getValue(w[3]), "fval" => getObjectiveValue(m))

	end

	function table_JuMP()


a=[0.5;1.0;5.0]
df = DataFrame(a=[0.5;1.0;5.0],c = zeros(3),omega1=zeros(3),omega2=zeros(3),omega3=zeros(3),fval=zeros(3))

for i in 1:length(a)
optim = max_JuMP(a[i])
df[i,:c] = optim["c"]
df[i,:omega1] = optim["w1"]
df[i,:omega2] = optim["w2"]
df[i,:omega3] = optim["w3"]
df[i,:fval] = optim["fval"]
end
##other solution with zip ## does not work
#for (aindex, c, w1, w2, w3, fv) in zip(df[:a], df[:c], df[:omega1], df[:omega2], df[:omega3], df[:fval])
#optim = max_JuMP(aindex)
#c = optim["c"]
#w1 = optim["w1"]
#w2 = optim["w2"]
#w3 = optim["w3"]
#fv = optim["fval"]

return df

	end

	
	function obj(x::Vector,grad::Vector,data::Dict)
# the x 4-component-vector should be the vector of weight of your different portfolio
c = x[1] # a scalar
y = x[2:end] # a 3x1 vector
a = data["a"]
u(z) = - exp(-a.*z)
# future payoff of the given portfolio
fx = data["payoff"]*y  #gives a (16x1) vector

if length(grad) > 0
	grad[1] = -a*u(c)
	grad[2:end] = -a.*(data["payoff"].*u(fx))'*data["pi"]
end 
return (u(c) + (data["pi"]'*u(fx))[1]) # the last term compute the expected payoff according to the states
# equivalent to - exp(-d["a"]*(x[1])) - (d["pi"]'*exp(-d["a"]*(d["payoff"]*x[2:end])))[1]
	end

	function constr(x::Vector,grad::Vector,data::Dict)
c = x[1]
y = x[2:end]
p = data["price"]
e = data["endow"]
if length(grad) > 0
	grad[1] = 1
	grad[2:end] = p 
end 
return c + (p'*(y.-e))[1]
# equivalently x[1] + d["price"]'*(x[2:end]- d["endow"])
	end

#first check obj([1,-1.41,0.8,1.6], grad, d)

	function max_NLopt(a=0.5)

# define an Opt object: which algorithm, how many dims of choice
opt = Opt(:LD_SLSQP, 4)
# set bounds and tolerance
upper_bounds!(opt,[+Inf for i=1:4])
lower_bounds!(opt,[0;[-Inf for i=1:3]])
xtol_rel!(opt,1e-4)

# define objective function
max_objective!(opt, (x,g) -> obj(x,g,data(a)))
# define constraints
# notice the anonymous function
equality_constraint!(opt, (x,g) -> constr(x,g,data(a)), 1e-5)
ftol_rel!(opt,1e-9)
# call optimize
(maxobj,maxval,ret) = optimize(opt, [0.,0.,0.,0.])

	end

	function table_NLopt()

	#should we do a similar dictionary?
# Dict("a" => a, "c" => getValue(c), "w1" => getValue(w[1]), "w2" => getValue(w[2]), "w3" => getValue(w[3]), "fval" => getObjectiveValue(m))
	
a=[0.5;1.0;5.0]
df = DataFrame(a=[0.5;1.0;5.0],c = zeros(3),omega1=zeros(3),omega2=zeros(3),omega3=zeros(3),fval=zeros(3))

for i in 1:length(a)
optim = max_NLopt(a[i])
df[i,1]
for j in 1:4
df[i,j+1] = optim[2][j]
df[i,end] = optim[1]
end
end

return df

	end

	# function `f` is for the NLopt interface, i.e.
	# it has 2 arguments `x` and `grad`, where `grad` is
	# modified in place
	# if you want to call `f` with more than those 2 args, you need to
	# specify an anonymous function as in
	# other_arg = 3.3
	# test_finite_diff((x,g)->f(x,g,other_arg), x )
	# this function cycles through all dimensions of `f` and applies
	# the finite differencing to each. it prints some nice output.
	function test_finite_diff(f::Function,x::Vector{Float64},tol=1e-6)

grad = zeros(length(x))
y = f(x,grad)
fdiff = finite_diff(x->f(x,grad),x)
euclidiandiff = sqrt(sum((fdiff - grad) .^ 2))
absdiff = find(abs(grad-fdiff))
if euclidiandiff > tol
	println("Differentiation test fails")
return (false, euclidiandiff)
else
println("Differentiation test passed!")
return true
end
 

	end

	# do this for each dimension of x
	# low-level function doing the actual finite difference
	function finite_diff(f::Function,x::Vector)

derivative = []
y = zeros(length(x))
for i in 1:length(x)
y[:] = x[:]
y[i] += sqrt(eps())
push!(derivative, (f(y) - f(x))/sqrt(eps()))
end

return derivative

	end

	function runAll()
		println("running tests:")
		include("test/runtests.jl")
		println("")
		println("JumP:")
		table_JuMP()
		println("")
		println("NLopt:")
		table_NLopt()
		ok = input("enter y to close this session.")
		if ok == "y"
			quit()
		end
	end


end


