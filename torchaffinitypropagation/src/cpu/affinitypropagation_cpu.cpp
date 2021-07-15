#include "affinitypropagation_cpu.h"
#include <stdio.h>

bool updateResponsibility(at::Tensor similarities, at::Tensor responsibilities, at::Tensor availabilities, int64_t sq_dim)
{
    auto avail_simi_sums = at::add(availabilities, similarities);
    auto  topkResultTuple = avail_simi_sums.topk((int64_t)2, (int64_t)1, true, true);
    auto topKValues = std::get<0>(topkResultTuple);
    auto topKIndices = std::get<1>(topkResultTuple);

    bool changed = false;

    for(int64_t i  = 0; i < sq_dim; i++)
    {
        auto topVal = topKValues[i][0].item();
        auto top2Val = topKValues[i][1].item();
        auto topIdx = topKIndices[i][0].item<int64_t>();

        for(int64_t k = 0; k < sq_dim; k++)
        {
            if(k != topIdx)
            {
                if(!changed && ((similarities[i][k] - topVal) != responsibilities[i][k]).item<bool>())
                {
                    changed = true;
                }
                responsibilities[i][k] = similarities[i][k] - topVal;
            }
            else
            {
                if(!changed && ((similarities[i][k] - top2Val) != responsibilities[i][k]).item<bool>())
                {
                    changed = true;
                }
                responsibilities[i][k] = similarities[i][k] - top2Val;
            }
        }
    }
    return changed;
}

bool updateAvailability(at::Tensor similarities, at::Tensor responsibilities, at::Tensor availabilities, int64_t sq_dim)
{
    bool changed = false;
    for(int64_t i  = 0; i < sq_dim; i++)
    {
        for(int64_t k = 0; k < sq_dim; k++)
        {
            // updating the non-self availabilities
            if(i != k)
            {
                auto self_resp = responsibilities[k][k];
                at::Tensor i_primes = torch::zeros({sq_dim - 2}, torch::TensorOptions().dtype(at::kLong));
                auto tmp = torch::arange(sq_dim, torch::TensorOptions().dtype(at::kLong));

                if(i < k)
                {
                    i_primes.index({at::indexing::Slice((long)0, (long)i, 1)}) = tmp.index({at::indexing::Slice((long)0, (long)i, 1)});
                    i_primes.index({at::indexing::Slice((long)i, (long)k-1, 1)}) = tmp.index({at::indexing::Slice((long)i+1, (long)k, 1)});
                    i_primes.index({at::indexing::Slice((long)k-1, (long)sq_dim-2, 1)}) = tmp.index({at::indexing::Slice((long)k+1, (long)sq_dim, 1)});
                }
                else
                {
                    i_primes.index({at::indexing::Slice((long)0, (long)k, 1)}) = tmp.index({at::indexing::Slice((long)0, (long)k, 1)});
                    i_primes.index({at::indexing::Slice((long)k, (long)i-1, 1)}) = tmp.index({at::indexing::Slice((long)k+1, (long)i, 1)});
                    i_primes.index({at::indexing::Slice((long)i-1, (long)sq_dim-2, 1)}) = tmp.index({at::indexing::Slice((long)i+1, (long)sq_dim, 1)});
                }
                auto to_add_candidates = responsibilities.index({i_primes, k});
                to_add_candidates = torch::maximum(to_add_candidates, torch::zeros_like(to_add_candidates));
                auto to_change = self_resp + torch::sum(to_add_candidates);
                to_change = torch::minimum(to_change, torch::zeros_like(to_change));
                if(!changed && (to_change != availabilities[i][k]).item<bool>())
                {
                    changed = true;
                }
                availabilities[i][k] = to_change;
            }

            // updating the self-availabilities
            else
            {
                
                at::Tensor i_primes = torch::zeros({sq_dim - 1}, torch::TensorOptions().dtype(at::kLong));
                auto tmp = torch::arange(sq_dim, torch::TensorOptions().dtype(at::kLong));
                i_primes.index({at::indexing::Slice(0, k, 1)}) = tmp.index({at::indexing::Slice(0, k, 1)});
                i_primes.index({at::indexing::Slice(k, sq_dim-1, 1)}) = tmp.index({at::indexing::Slice(k+1, sq_dim, 1)});
                
                auto to_add_candidates = responsibilities.index({i_primes, k});
                to_add_candidates = torch::maximum(to_add_candidates, torch::zeros_like(to_add_candidates));
                auto to_change = torch::sum(to_add_candidates);
                if(!changed && (to_change != availabilities[k][k]).item<bool>())
                {
                    changed = true;
                }
                availabilities[k][k] = to_change;
            }
        }
    }
    return changed;
}

std::vector<at::Tensor> affinitypropagation_cpu(at::Tensor similarities, int64_t iters)
{
    // Get the dimensions
    auto nrow = similarities.size(0);
    auto ncol = similarities.size(1);
    std::string msg = "rows :";
    msg += std::to_string(nrow);
    msg += ", cols:";
    msg += std::to_string(ncol);
    msg += "\n";
    std::cout << msg.c_str() << std::endl;
    
    if(nrow != ncol)
    {
        std::string errMsg = "Similarity matrix must be square! Got row/col counts of ";
        errMsg += std::to_string(nrow);
        errMsg += ",";
        errMsg += std::to_string(ncol);
        throw std::invalid_argument(errMsg.c_str());
    }

    auto responsibilities = torch::zeros_like(similarities);
    auto availabilities = torch::zeros_like(similarities);


    AT_DISPATCH_ALL_TYPES(similarities.type(), "affinity propagation cpu", [&] {
        bool changed = true;
        for(auto i = 0; (i < iters || iters == -1) && changed; i++)
        {
            changed &= updateResponsibility(similarities, responsibilities, availabilities, nrow);
            changed &= updateAvailability(similarities, responsibilities, availabilities, nrow);
        }
    });

    return {responsibilities, availabilities};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) 
{
    m.def("affinitypropagation_cpu", &affinitypropagation_cpu, "affinity propagation (CPU)");
}

