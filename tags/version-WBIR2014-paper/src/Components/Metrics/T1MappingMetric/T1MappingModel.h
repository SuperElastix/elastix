#ifndef T1MAPPINGMODEL_H
#define T1MAPPINGMODEL_H
#include "ModelBaseClass.h"
#include <vector>

class T1MappingModel : public ModelBaseClass
{
public:
    T1MappingModel ();
    virtual double Evaluate (double x,  std::vector< double >  p);
    virtual std::vector<double> EvaluateDerivative (double x,  std::vector< double >  p);
};
#endif //T1MAPPINGMODEL_H

