#include <iostream>
#include "vnl/vnl_diag_matrix.h"
#include "vnl/algo/vnl_determinant.h"
#include "LMOptimizer.h"
#include "ModelBaseClass.h"

LMOptimizer::LMOptimizer()
{
   //Member variables
        m_model = NULL;
        m_lambda = 0.0;
        m_scale = 10.0;
}

LMOptimizer::~LMOptimizer()
{
    if(NULL != m_model )
    {
        delete m_model;
    }
}

void LMOptimizer::SetDampingParameter( double lambda )
{
    m_lambda = lambda;
}

void LMOptimizer::SetModel(ModelBaseClass * model)
{
    m_model = model;
}

bool LMOptimizer::SetValues( VectorType x, VectorType y )
{
    if( x.size() == y.size() )
    {
        m_x = x;
        m_y = y;
        return true;
    }
    else
    {
        std::cerr << "bool LMOptimizer::SetValues  sizes of input are not equal" << std::endl;
        return false;
    }
}

void LMOptimizer::UpdateDiff( VectorType params, vnl_vector<double> & diff )
{
    for( unsigned int xIndex = 0; xIndex < m_x.size(); ++xIndex )
    {
        diff[ xIndex ] = m_model->Evaluate( m_x[ xIndex ], params ) -  m_y[ xIndex ];
    }
}

void LMOptimizer::UpdateJacobian ()
{
    for( unsigned int xIndex = 0; xIndex < m_x.size(); ++xIndex )
    {
        double x = this->m_x[ xIndex ];
       std::vector< double > dSdC = m_model->EvaluateDerivative( x, m_Parameters );
        for( unsigned int pIndex = 0; pIndex < m_Parameters.size(); ++pIndex )
        {
            m_Jacobian(xIndex, pIndex ) = 2.0*m_diff[ xIndex ]*dSdC[ pIndex ];
        }
    }
}

vnl_vector< double > LMOptimizer::CalculateStep( )
{
    m_Hessian = m_Jacobian.transpose() * m_Jacobian;
    vnl_diag_matrix < double > diagH(m_Hessian.get_diagonal());
    diagH *= m_lambda;
    MatrixType Hinv( m_Parameters.size(), m_Parameters.size());
    vnl_vector< double > delta;
    delta.set_size( m_Parameters.size() );
    if(vnl_determinant( m_Hessian + diagH) > 1e-5 )
    {
        Hinv = vnl_inverse( m_Hessian + diagH );
        vnl_vector< double > mdiffsquared( m_diff.size() );
        for(unsigned int i = 0; i < m_diff.size(); ++i)
        {
            mdiffsquared[ i ] = m_diff[ i ] * m_diff[ i ];
        }
        delta = Hinv * ( m_Jacobian.transpose()  *  mdiffsquared );
    }
    else
    {
        delta.fill( 0.0 );
    }

    return delta;
}

void LMOptimizer::Run( unsigned int nrOfIterations, VectorType & initParams )
{
    m_Parameters = initParams;
    VectorType bufferParams( m_Parameters.size() );

    m_Parameters = initParams;
    m_Jacobian.set_size(m_x.size(), m_Parameters.size());
    m_diff.set_size( m_x.size() );
    UpdateDiff( m_Parameters, m_diff );
    UpdateJacobian( );

    m_initValues.resize( m_x.size() );
    for(unsigned int i = 0; i < m_x.size(); ++i)
    {
        m_initValues[ i ] = m_model->Evaluate(m_x[ i ], m_Parameters );
    }

    double err0 = m_diff.squared_magnitude();

    for(unsigned int ii = 0; ii < nrOfIterations; ++ii )
    {
        vnl_vector< double > delta = CalculateStep( ); // Calculate step

        for(unsigned int jj= 0;jj < m_Parameters.size(); ++jj)
        {
            bufferParams[ jj ] = m_Parameters[ jj ] - delta[ jj ];
        }
        UpdateDiff( bufferParams, m_diff );

        double err = m_diff.squared_magnitude();

        if(err < err0)
        {
            SetDampingParameter( m_lambda / m_scale );
            err0 = err;
            m_Parameters  = bufferParams;
            UpdateJacobian();
        }
        else
        {
            //SetDampingParameter( m_lambda *  m_scale );
            break;
        }
    }

    m_Err = err0;
    m_values.resize( m_x.size() );
    for(unsigned int i = 0; i < m_x.size(); ++i)
    {
        m_values[ i ] = m_model->Evaluate(m_x[ i ], m_Parameters );
    }

    initParams = m_Parameters;
}

double LMOptimizer::GetErr()
{
    return m_Err;
}

void LMOptimizer::Print( const VectorType & vec )
{
    for( std::vector<double>::const_iterator i = vec.begin(); i != vec.end(); ++i)
        std::cout << *i << ' ';
    std::cout << std::endl;
}
LMOptimizer::VectorType LMOptimizer::GetValues()
{
    return m_values;
}

LMOptimizer::VectorType LMOptimizer::GetInitValues()
{
    return m_initValues;
}

LMOptimizer::VectorType LMOptimizer::VnlToStdVector( vnl_vector< double > vec )
{
    VectorType stdvec( vec.size() );
    for(unsigned int i = 0; i < vec.size(); ++i )
    {
        stdvec[ i ] = vec[ i ];
    }

    return stdvec;
}

