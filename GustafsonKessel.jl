module GustafsonKessel
	using LinearAlgebra, Statistics
	export gustafson_kessel

	function gustafson_kessel(X::Matrix{T}, C::Int; α=2.0, ρ = ones(C), 
		maxiter=100, tol=1e-5, stabilize=true, γ=0.1, β=10^15) where {T <: Real}

		α > 1 || throw(ArgumentError("α must be greater than 1"))
		size(ρ, 1) == C || throw(ArgumentError("length of ρ must equal to C"))

		N = size(X, 2)
		d = size(X, 1)
		W = rand(float(T), N, C)
		W ./= sum(W, dims=2)
		distances = zeros(N, C)
		e = 2 / (α - 1)
		J = Inf
		P0_det = stabilize ? det(cov(X')) ^(1/d) : 0
		
		V = Matrix{Float64}(undef, d, C)

		for iter in 1:maxiter
			pow_W = W .^ α
			W_sum = sum(pow_W, dims=1)
			V .= (X * pow_W) ./ W_sum
			for (i, vi) in enumerate(eachcol(V))
				P = zeros(d, d)
				for(j, xj) in enumerate(eachcol(X))
					sub = xj - vi
					P += pow_W[j, i] .* sub * sub'
				end
				
				P ./= W_sum[1, i]
				stabilize && _recalculate_cov_matrix(P, γ, β, P0_det)
				A = (ρ[i] * det(P))^(1/d) * inv(P)

				for(j, xj) in enumerate(eachcol(X))
					sub = xj - vi
					distances[j, i] = sub' * A * sub
				end
			end

			J_new = 0
			for (i, di) in enumerate(eachrow(distances))
				idx = findfirst(==(0), di)

				if idx === nothing
					for (j, dij) in enumerate(di)
						denom = sum((dij ./ di[k]) .^ e for k in 1:C)
						W[i, j] = 1/denom
						J_new += W[i, j] * dij
					end
				else
					result = zeros(N)
					result[idx] = 1
					W[i, :] .= result
				end
			end

			abs(J - J_new) < tol && break
			J = J_new
		end

		return V, W
	end

	function _recalculate_cov_matrix(P::Matrix{Float64}, γ::Number,
									β::Int, P0_det::Float64)
		temp_P = (1 - γ) * P + γ * P0_det * I(size(P, 1))
		eig = eigen(Symmetric(temp_P))
		eig_vals = eig.values
		max_val = maximum(eig_vals)

		eig_vals[max_val ./ eig_vals .> β] .= max_val / β 

		P .= eig.vectors * diagm(eig_vals) * eig.vectors'
	end
end