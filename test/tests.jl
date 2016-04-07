



module AssetTests

	using FactCheck, HW_constrained

	results_jump = HW_constrained.table_JuMP()
	results_NLopt = HW_constrained.table_NLopt()
	results_true = data()["truth"]
	tol = 1e-8
	x = [1.0,1.0,1.0,1.0]
	grad = zeros(4)
	
	context("testing components") do

		facts("finite differences") do

		end

		facts("test_finite_diff") do

		end

		facts("tests gradient of objective function") do

			@fact HW_constrained.test_finite_diff((x, grad)->HW_constrained.obj(x, grad, HW_constrained.data()),x,tol) --> true

		end


		facts("tests gradient of constraint function") do

			@fact HW_constrained.test_finite_diff((x, grad)->HW_constrained.constr(x, grad, HW_constrained.data()),x,tol) --> true

		end

	end

	context("testing result of both maximization methods with a $tol tolerance") do

results_jump = HW_constrained.table_JuMP()
results_NLopt = HW_constrained.table_NLopt()
results_true = HW_constrained.data()["truth"]


		facts("testing results for Jump maximization method with a $tol tolerance") do

			@fact results_jump[:c] --> roughly(results_true[:c]; atol=tol)
			@fact results_jump[:omega1] --> roughly(results_true[:omega1]; atol=tol)
			@fact results_jump[:omega2] --> roughly(results_true[:omega2]; atol=tol)
			@fact results_jump[:omega3] --> roughly(results_true[:omega3]; atol=tol)
			@fact results_jump[:fval] --> roughly(results_true[:fval]; atol=tol)

		end

		facts("testing results for NLopt maximization method with a $tol tolerance") do

			@fact results_NLopt[:c] --> roughly(results_true[:c]; atol=tol)
			@fact results_NLopt[:omega1] --> roughly(results_true[:omega1]; atol=tol)
			@fact results_NLopt[:omega2] --> roughly(results_true[:omega2]; atol=tol)
			@fact results_NLopt[:omega3] --> roughly(results_true[:omega3]; atol=tol)
			@fact results_NLopt[:fval] --> roughly(results_true[:fval]; atol=tol)

		end

	end




end
