

function invl(Γl′::AbstractTensorMap, λlu::AbstractTensorMap, λll::AbstractTensorMap, λld::AbstractTensorMap)
    iλlu, iλll, iλld = map(x -> inv(x),(λlu, λll, λld))
    @tensor Γl′[-1,-2,-3;-4,-5] = iλlu[-2,2] * iλll[5,-5] * iλld[4,-4] * Γl′[-1,2,-3,4,5]
    return Γl′
end

function invr(Γr′::AbstractTensorMap, λrr::AbstractTensorMap, λru::AbstractTensorMap, λrd::AbstractTensorMap)
    iλrr, iλru, iλrd = map(x -> inv(x),(λrr, λru, λrd))
    @tensor Γr′[-1,-2,-3;-4,-5] = iλrr[-1,1] * iλru[-2,2] * iλrd[4,-4] * Γr′[1,2,-3,4,-5]
    return Γr′
end

function invu(Γu′::AbstractTensorMap, λur::AbstractTensorMap, λuu::AbstractTensorMap, λul::AbstractTensorMap)
    iλur, iλuu, iλul = map(x -> inv(x),(λur, λuu, λul))
    @tensor Γu′[-1,-2,-3;-4,-5] = iλur[-1,1] * iλuu[-2,2] * iλul[5,-5] * Γu′[1,2,-3,-4,5]
    return Γu′
end
function invd(Γd′::AbstractTensorMap, λdr::AbstractTensorMap, λdd::AbstractTensorMap, λdl::AbstractTensorMap)
    iλdr, iλdd, iλdl = map(x -> inv(x),(λdr, λdd, λdl))
    @tensor Γd′[-1,-2,-3;-4,-5] = iλdr[-1,1] * iλdd[4,-4] * iλdl[5,-5] * Γd′[1,-2,-3,4,5]
    return Γd′
end

function Γcontractlr(Γl′::AbstractTensorMap, Γr′::AbstractTensorMap)
    @tensor tmp[-1,-2,-3,-4,-5;-6,-7,-8] ≔ Γl′[1,-3,-5,-7,-8] * Γr′[-1,-2,-4,-6,1]
    return tmp
end

function Γcontractud(Γd′::AbstractTensorMap,Γu′::AbstractTensorMap)
    @tensor tmp[-1,-2,-3,-4,-5;-6,-7,-8] ≔ Γu′[-2,-3,-5,1,-8] * Γd′[-1,1,-4,-6,-7]
    return tmp
end

function actionlr(tmp::AbstractTensorMap, O::AbstractTensorMap)
    @tensor tmp′[-1,-2,-3,-4,-5;-6,-7,-8] ≔ tmp[-1,-2,-3,1,2,-6,-7,-8] * O[-5,-4,2,1]
    return tmp′
end

function actionud(tmp::AbstractTensorMap, O::AbstractTensorMap)
    @tensor tmp′[-1,-2,-3,-4,-5;-6,-7,-8] ≔ tmp[-1,-2,-3,1,2,-6,-7,-8] * O[-4,-5,1,2]
    return tmp′
end

function λΓcontract(Γ::AbstractTensorMap, 
    λlr::AbstractTensorMap, λlu::AbstractTensorMap, λld::AbstractTensorMap, λll::AbstractTensorMap)
    @tensor Γl′[-1,-2,-3;-4,-5] ≔ λlr[-1,1] * λlu[-2,2] * λll[5,-5] * λld[4,-4] * Γ[1,2,-3,4,5]
    return Γl′
end