#ifndef LMOPTIMIZER_H
#define LMOPTIMIZER_H
#include <vector>
#include "vnl/algo/vnl_matrix_update.h"
#include "vnl/vnl_inverse.h"

class ModelBaseClass;
class LMOptimizer
{
        public: //Interface

    // Constructor
    LMOptimizer();

    //Destructor
        ~LMOptimizer();

    //typedefs
    typedef std::vector< double > VectorType;
    typedef vnl_matrix< double > MatrixType;

    //Functions & voids
    void Run( unsigned int, VectorType & );
    bool SetValues( VectorType x, VectorType y );

    //LMOptimizer will free memory of model
    void SetModel( ModelBaseClass *);
    void SetDampingParameter( double );
    void SetModelParameters( const VectorType & );
    VectorType GetValues();
    VectorType GetInitValues();
    double GetErr();
    void InitializeValues();
    void Print(const VectorType & );

private:

    //Input member parameters
    VectorType m_x;
    VectorType m_y;
    double m_lambda;
    double m_scale;
    double m_Err;
    vnl_vector< double >  m_diff;
    ModelBaseClass * m_model;
    MatrixType m_Jacobian;
    MatrixType m_Hessian;
    VectorType m_Parameters;
    VectorType m_values;
    VectorType m_initValues;
    double m_residuals;

    void UpdateJacobian ();
    vnl_vector< double > CalculateStep();
    void UpdateDiff(VectorType params, vnl_vector< double > & diff);
    VectorType VnlToStdVector( vnl_vector< double > vec );
    //void UpdateDiff( double val );
};
#endif //LMOPTIMIZER_H

