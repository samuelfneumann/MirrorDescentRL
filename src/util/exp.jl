module ExpUtils

import ChoosyDataLoggers: ChoosyDataLoggers, @data
import ChoosyDataLoggers: construct_logger, NotDataFilter, DataLogger

function ChoosyDataLoggers.construct_logger(; steps=nothing, extra_groups_and_names=[])
    return ChoosyDataLoggers.construct_logger([[:exp]; extra_groups_and_names]; steps=steps)
end

function prep_save_results(data, save_extras)
    save_results = copy(data[:exp])
    for ex in save_extras
        if ex isa AbstractArray
            save_results[Symbol(ex[1]*"_"*ex[2])] = data[Symbol(ex[1])][Symbol(ex[2])]
        else
            for k in keys(data[Symbol(ex)])
                save_results[Symbol(ex * "_" * string(k))] = data[Symbol(ex)][Symbol(k)]
            end
        end
    end
    return Dict{String,Any}(
        [
            k => v for (k, v) in zip(string.(keys(save_results)), values(save_results))
        ],
    )
end

end # ExpUtils end
