#ifndef MODELBASECLASS_H
#define MODELBASECLASS_H
#include <vector>

class ModelBaseClass
{
	public:
    ModelBaseClass();
        virtual double Evaluate (double x,  std::vector< double > p) = 0;
        virtual std::vector< double > EvaluateDerivative (double x,  std::vector< double >  p) = 0;
};
#endif // MODELBASECLASS_H
