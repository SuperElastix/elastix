#ifndef _itkLinearScalingGroupwiseMetric_HXX__
#define _itkLinearScalingGroupwiseMetric_HXX__

#include "itkLinearScalingGroupwiseMetric.h"

#ifdef ELASTIX_USE_OPENMP
#include <omp.h>
#endif

namespace itk
{
    /**
     * ********************* Constructor ******************************
     */
    
    template< class TFixedImage, class TMovingImage >
    LinearScalingGroupwiseMetric< TFixedImage, TMovingImage >
    ::LinearScalingGroupwiseMetric()
    {

    }
    
    /**
     * ******************* Destructor *******************
     */
    
    template< class TFixedImage, class TMovingImage >
    LinearScalingGroupwiseMetric< TFixedImage, TMovingImage >
    ::~LinearScalingGroupwiseMetric()
    {
    }
    
    /**
     * ******************* Initialize *******************
     */
    
    template< class TFixedImage, class TMovingImage >
    void
    LinearScalingGroupwiseMetric< TFixedImage, TMovingImage >
    ::Initialize( void ) throw ( ExceptionObject )
    {
        this->Superclass::Initialize();
        
        const unsigned int lastDim     = this->GetFixedImage()->GetImageDimension() - 1;
        const unsigned int lastDimSize = this->GetFixedImage()->GetLargestPossibleRegion().GetSize( lastDim );

        this->m_RandomList.assign(lastDimSize, 0.0);
    }
    
    /**
     * ********************* GetValue ****************************
     */
    
    template< class TFixedImage, class TMovingImage >
    typename LinearScalingGroupwiseMetric< TFixedImage, TMovingImage >::MeasureType
    LinearScalingGroupwiseMetric< TFixedImage, TMovingImage >
    ::GetValue( const ParametersType & parameters ) const
    {
        const unsigned int lastDim     = this->GetFixedImage()->GetImageDimension() - 1;
        const unsigned int lastDimSize = this->GetFixedImage()->GetLargestPossibleRegion().GetSize( lastDim );

        MeasureType value;
        
        std::vector<MeasureType> values;

        bool minimize =true;
        
        this->GetValue(parameters, values, minimize);
        
        this->m_MetricCombiner->Combine(value, values, lastDimSize, minimize);
        
        return value;
    }
    
    /**
     * ********************* GetDerivative ****************************
     */
    
    template< class TFixedImage, class TMovingImage >
    void
    LinearScalingGroupwiseMetric< TFixedImage, TMovingImage >
    ::GetDerivative(const ParametersType & parameters, DerivativeType & derivative ) const
    {
        MeasureType value;
        this->GetValueAndDerivative( parameters, value, derivative );
    }
    
    
    /**
     * *********************  GetValueAndDerivative ****************************
     */
    
    template< class TFixedImage, class TMovingImage >
    void
    LinearScalingGroupwiseMetric< TFixedImage, TMovingImage >
    ::GetValueAndDerivative( const ParametersType & parameters, MeasureType & value, DerivativeType & derivative ) const
    {
        const unsigned int lastDim     = this->GetFixedImage()->GetImageDimension() - 1;
        const unsigned int lastDimSize = this->GetFixedImage()->GetLargestPossibleRegion().GetSize( lastDim );

        std::vector<MeasureType> values;
        std::vector<DerivativeType> derivatives;
        
        bool minimize =true;
        
        this->GetValueAndDerivative(parameters, values, derivatives, minimize);
        
        this->m_MetricCombiner->Combine(value, derivative, values, derivatives, lastDimSize, minimize);
        
        if( this->m_SubtractMean )
        {
            if( !this->m_TransformIsStackTransform )
            {
                const unsigned int lastDimGridSize              = this->m_GridSize[ lastDim ];
                const unsigned int numParametersPerDimension    = this->GetNumberOfParameters() / this->GetMovingImage()->GetImageDimension();
                const unsigned int numControlPointsPerDimension = numParametersPerDimension / lastDimGridSize;
                DerivativeType     mean( numControlPointsPerDimension );
                for( unsigned int d = 0; d < this->GetMovingImage()->GetImageDimension(); ++d )
                {
                    mean.Fill( 0.0 );
                    const unsigned int starti = numParametersPerDimension * d;
                    for( unsigned int i = starti; i < starti + numParametersPerDimension; ++i )
                    {
                        const unsigned int index = i % numControlPointsPerDimension;
                        mean[ index ] += derivative[ i ];
                    }
                    mean /= static_cast< double >( lastDimGridSize );
                    
                    for( unsigned int i = starti; i < starti + numParametersPerDimension; ++i )
                    {
                        const unsigned int index = i % numControlPointsPerDimension;
                        derivative[ i ] -= mean[ index ];
                    }
                }
            }
            else
            {
                const unsigned int numParametersPerLastDimension = this->GetNumberOfParameters() / lastDimSize;
                DerivativeType     mean( numParametersPerLastDimension );
                mean.Fill( 0.0 );
                
                for( unsigned int t = 0; t < lastDimSize; ++t )
                {
                    const unsigned int startc = numParametersPerLastDimension * t;
                    for( unsigned int c = startc; c < startc + numParametersPerLastDimension; ++c )
                    {
                        const unsigned int index = c % numParametersPerLastDimension;
                        mean[ index ] += derivative[ c ];
                    }
                }
                mean /= static_cast< double >( lastDimSize );
                
                for( unsigned int t = 0; t < lastDimSize; ++t )
                {
                    const unsigned int startc = numParametersPerLastDimension * t;
                    for( unsigned int c = startc; c < startc + numParametersPerLastDimension; ++c )
                    {
                        const unsigned int index = c % numParametersPerLastDimension;
                        derivative[ c ] -= mean[ index ];
                    }
                }
            }
            
        }
        
    }
    
    /**
     * ******************* SampleRandom *******************
     */
    
    template< class TFixedImage, class TMovingImage >
    void
    LinearScalingGroupwiseMetric< TFixedImage, TMovingImage >
    ::SampleRandom( const int n, const int m, std::vector< int > & numbers ) const
    {
        numbers.clear();
        
        Statistics::MersenneTwisterRandomVariateGenerator::Pointer randomGenerator
        = Statistics::MersenneTwisterRandomVariateGenerator::GetInstance();
        
        for( int i = 0; i < n; ++i )
        {
            int randomNum = 0;
            do
            {
                randomNum = static_cast< int >( randomGenerator->GetVariateWithClosedRange( m ) );
            }
            while( find( numbers.begin(), numbers.end(), randomNum ) != numbers.end() );
            numbers.push_back( randomNum );
        }
    } // end SampleRandom()
    
    /**
     * *************** EvaluateTransformJacobianInnerProduct ****************
     */
    
    template< class TFixedImage, class TMovingImage >
    void
    LinearScalingGroupwiseMetric< TFixedImage, TMovingImage >
    ::EvaluateTransformJacobianInnerProduct(const TransformJacobianType & jacobian, const MovingImageDerivativeType & movingImageDerivative, DerivativeType & imageJacobian ) const
    {
        typedef typename TransformJacobianType::const_iterator JacobianIteratorType;
        typedef typename DerivativeType::iterator              DerivativeIteratorType;
        JacobianIteratorType jac = jacobian.begin();
        imageJacobian.Fill( 0.0 );
        const unsigned int sizeImageJacobian = imageJacobian.GetSize();
        for( unsigned int dim = 0; dim < FixedImageDimension; dim++ )
        {
            const double           imDeriv = movingImageDerivative[ dim ];
            DerivativeIteratorType imjac   = imageJacobian.begin();
            
            for( unsigned int mu = 0; mu < sizeImageJacobian; mu++ )
            {
                ( *imjac ) += ( *jac ) * imDeriv;
                ++imjac;
                ++jac;
            }
        }
    } // end EvaluateTransformJacobianInnerProduct()
    
    /**
     * ***************** SetTemplateToAA ***********************
     */
    
    template< class TFixedImage, class TMovingImage >
    void
    LinearScalingGroupwiseMetric< TFixedImage, TMovingImage >
    ::SetTemplateToAA( void )
    {
        this->m_TemplateImage = new AATemplateType;
    }
    
    /**
     * ***************** SetTemplateToGA ***********************
     */
    
    template< class TFixedImage, class TMovingImage >
    void
    LinearScalingGroupwiseMetric< TFixedImage, TMovingImage >
    ::SetTemplateToGA( void )
    {
        this->m_TemplateImage = new GATemplateType;
    }
    
    /**
     * ***************** SetTemplateToHA ***********************
     */
    
    template< class TFixedImage, class TMovingImage >
    void
    LinearScalingGroupwiseMetric< TFixedImage, TMovingImage >
    ::SetTemplateToHA( void )
    {
        this->m_TemplateImage = new HATemplateType;
    }
    
    /**
     * ***************** SetCombinationToSum ***********************
     */
    
    template< class TFixedImage, class TMovingImage >
    void
    LinearScalingGroupwiseMetric< TFixedImage, TMovingImage >
    ::SetCombinationToSum( void )
    {
        this->m_MetricCombiner = new MetricSumType;
    }
    
    /**
     * ***************** SetCombinationToSqSum ***********************
     */
    
    template< class TFixedImage, class TMovingImage >
    void
    LinearScalingGroupwiseMetric< TFixedImage, TMovingImage >
    ::SetCombinationToSqSum( void )
    {
        this->m_MetricCombiner = new MetricSquaredSumType;
    }

    
}

#endif // end #ifndef _itkLinearScalingGroupwiseMetric_HXX__
