#include <math.h>
#include "ModelBaseClass.h"
#include "T1MappingModel.h"

T1MappingModel::T1MappingModel()
    : ModelBaseClass ()
{

}

double T1MappingModel::Evaluate (double x,  std::vector< double >  p)
{
    return  p[ 0 ] * exp (-x*p[ 1 ] ) + p[ 2 ];
}

std::vector<double> T1MappingModel::EvaluateDerivative (double x, std::vector< double > p)
{
    std::vector< double > dp( p.size() );
    dp[ 0 ] = exp(- x*p[ 1 ]);
    dp[ 1 ] = -p[ 0 ] * x * exp(-x*p[ 1 ]);
    dp[ 2 ] = 1.0;

    return dp;
}
