#ifndef __itkStackTransformBendingEnergyPenaltyTerm_hxx
#define __itkStackTransformBendingEnergyPenaltyTerm_hxx

#include "itkStackTransformBendingEnergyPenaltyTerm.h"

#ifdef ELASTIX_USE_OPENMP
#include <omp.h>
#endif

namespace itk
{

/**
 * ****************** Constructor *******************************
 */

template< class TFixedImage, class TScalarType >
StackTransformBendingEnergyPenaltyTerm< TFixedImage, TScalarType >
::StackTransformBendingEnergyPenaltyTerm()
{

  this->SetUseImageSampler( true );

} // end Constructor


/**
 * ****************** GetValue *******************************
 */

template< class TFixedImage, class TScalarType >
typename StackTransformBendingEnergyPenaltyTerm< TFixedImage, TScalarType >::MeasureType
StackTransformBendingEnergyPenaltyTerm< TFixedImage, TScalarType >
::GetValue( const ParametersType & parameters ) const
{
    this->m_NumberOfPixelsCounted = 0;
    RealType           measure = NumericTraits< RealType >::Zero;
    SpatialHessianType spatialHessian;
    if( !this->m_AdvancedTransform->GetHasNonZeroSpatialHessian() )
    {
        return static_cast< MeasureType >( measure );
    }

    if(this->m_SampleLastDimensionRandomly)
    {
        this->m_RandomList.resize(this->m_NumSamplesLastDimension);
        this->SampleRandom(this->GetFixedImage()->GetLargestPossibleRegion().GetSize( ReducedFixedImageDimension ),this->m_NumSamplesLastDimension,this->m_RandomList);
    }
    else
    {
        this->m_NumSamplesLastDimension = this->GetFixedImage()->GetLargestPossibleRegion().GetSize(ReducedFixedImageDimension);
        this->m_RandomList.resize(this->m_NumSamplesLastDimension);
        for( unsigned int d = 0; d < this->m_NumSamplesLastDimension; d++ )
        {
            this->m_RandomList[d]=d;
        }
        
    }

    this->BeforeThreadedGetValueAndDerivative( parameters );
    
    ImageSampleContainerPointer sampleContainer = this->GetImageSampler()->GetOutput();
    
    typename ImageSampleContainerType::ConstIterator fiter;
    typename ImageSampleContainerType::ConstIterator fbegin = sampleContainer->Begin();
    typename ImageSampleContainerType::ConstIterator fend   = sampleContainer->End();

    for( fiter = fbegin; fiter != fend; ++fiter )
    {
        std::vector< FixedImagePointType >  fixedPoints(this->m_NumSamplesLastDimension);
        
        FixedImagePointType fixedPoint = ( *fiter ).Value().m_ImageCoordinates;
        FixedImageContinuousIndexType voxelCoord;
        this->GetFixedImage()->TransformPhysicalPointToContinuousIndex( fixedPoint, voxelCoord );
        
        unsigned int       numSamplesOk            = 0;
        
        for( unsigned int s = 0; s < this->m_NumSamplesLastDimension; s++ )
        {
            RealType             movingImageValueTemp;
            MovingImagePointType mappedPoint;
            
            voxelCoord[ ReducedFixedImageDimension ] = this->m_RandomList[ s ];
            
            this->GetFixedImage()->TransformContinuousIndexToPhysicalPoint( voxelCoord, fixedPoint );
            
            bool sampleOk = this->TransformPoint( fixedPoint, mappedPoint );
            
            if( sampleOk )
            {
                sampleOk = this->IsInsideMovingMask( mappedPoint );
            }
            
            if( sampleOk )
            {
                fixedPoints[numSamplesOk] = fixedPoint;
                numSamplesOk++;
            }
            
        }
        
        if (numSamplesOk > 0)
        {
            this->m_NumberOfPixelsCounted++;
            fixedPoints.resize(numSamplesOk);
            for( unsigned int o = 0; o < numSamplesOk; o++ )
            {
                this->m_AdvancedTransform->GetSpatialHessian( fixedPoints[o], spatialHessian);
                
                for( unsigned int k = 0; k < ReducedFixedImageDimension; k++ )
                {
                    measure += vnl_math_sqr(spatialHessian[ k ].GetVnlMatrix().frobenius_norm() ) / numSamplesOk;
                }

            }
            
        }
    }

    this->CheckNumberOfSamples( sampleContainer->Size(), this->m_NumberOfPixelsCounted );
    measure /= static_cast< RealType >( this->m_NumberOfPixelsCounted );
    
    return static_cast< MeasureType >( measure );

} // end GetValue()


/**
 * ******************* GetDerivative *******************
 */

template< class TFixedImage, class TScalarType >
void
StackTransformBendingEnergyPenaltyTerm< TFixedImage, TScalarType >
::GetDerivative(
  const ParametersType & parameters,
  DerivativeType & derivative ) const
{
  MeasureType dummyvalue = NumericTraits< MeasureType >::Zero;
  this->GetValueAndDerivative( parameters, dummyvalue, derivative );

} // end GetDerivative()


/**
 * ****************** GetValueAndDerivativeSingleThreaded *******************************
 */

template< class TFixedImage, class TScalarType >
void
StackTransformBendingEnergyPenaltyTerm< TFixedImage, TScalarType >
::GetValueAndDerivativeSingleThreaded(
  const ParametersType & parameters,
  MeasureType & value,
  DerivativeType & derivative ) const
{
    FixedImageSizeType m_GridSize;

    this->m_NumberOfPixelsCounted = 0;
    RealType           measure = NumericTraits< RealType >::Zero;
    derivative = DerivativeType( this->GetNumberOfParameters() );
    derivative.Fill( NumericTraits< DerivativeValueType >::ZeroValue() );
    
    SpatialHessianType           spatialHessian;
    JacobianOfSpatialHessianType jacobianOfSpatialHessian;
    NonZeroJacobianIndicesType   nonZeroJacobianIndices;
    const NumberOfParametersType numberOfNonZeroJacobianIndices
    = this->m_AdvancedTransform->GetNumberOfNonZeroJacobianIndices();
    jacobianOfSpatialHessian.resize( numberOfNonZeroJacobianIndices );
    nonZeroJacobianIndices.resize( numberOfNonZeroJacobianIndices );
    
    if( !this->m_AdvancedTransform->GetHasNonZeroSpatialHessian()
       && !this->m_AdvancedTransform->GetHasNonZeroJacobianOfSpatialHessian() )
    {
        value = static_cast< MeasureType >( measure );
        return;
    }
    
    if(this->m_SampleLastDimensionRandomly)
    {
        this->m_RandomList.resize(this->m_NumSamplesLastDimension);
        this->SampleRandom(this->GetFixedImage()->GetLargestPossibleRegion().GetSize( ReducedFixedImageDimension ),this->m_NumSamplesLastDimension,this->m_RandomList);
    }
    else
    {
        this->m_NumSamplesLastDimension = this->GetFixedImage()->GetLargestPossibleRegion().GetSize(ReducedFixedImageDimension);
        this->m_RandomList.resize(this->m_NumSamplesLastDimension);
        for( unsigned int d = 0; d < this->m_NumSamplesLastDimension; d++ )
        {
            this->m_RandomList[d]=d;
        }
        
    }
    
    this->BeforeThreadedGetValueAndDerivative( parameters );
    
    ImageSampleContainerPointer sampleContainer = this->GetImageSampler()->GetOutput();
    
    typename ImageSampleContainerType::ConstIterator fiter;
    typename ImageSampleContainerType::ConstIterator fbegin = sampleContainer->Begin();
    typename ImageSampleContainerType::ConstIterator fend   = sampleContainer->End();
    
    for( fiter = fbegin; fiter != fend; ++fiter )
    {
        std::vector< FixedImagePointType >  fixedPoints(this->m_NumSamplesLastDimension);
        
        FixedImagePointType fixedPoint = ( *fiter ).Value().m_ImageCoordinates;
        FixedImageContinuousIndexType voxelCoord;
        this->GetFixedImage()->TransformPhysicalPointToContinuousIndex( fixedPoint, voxelCoord );
        
        unsigned int       numSamplesOk            = 0;
        
        for( unsigned int s = 0; s < this->m_NumSamplesLastDimension; s++ )
        {
            RealType             movingImageValueTemp;
            MovingImagePointType mappedPoint;
            
            voxelCoord[ ReducedFixedImageDimension ] = this->m_RandomList[ s ];
            
            this->GetFixedImage()->TransformContinuousIndexToPhysicalPoint( voxelCoord, fixedPoint );
            
            bool sampleOk = this->TransformPoint( fixedPoint, mappedPoint );
            
            if( sampleOk )
            {
                sampleOk = this->IsInsideMovingMask( mappedPoint );
            }
            
            if( sampleOk )
            {
                fixedPoints[numSamplesOk] = fixedPoint;
                numSamplesOk++;

            }
            
        }
        if (numSamplesOk > 0)
        {
            this->m_NumberOfPixelsCounted++;
            fixedPoints.resize(numSamplesOk);
            for( unsigned int o = 0; o < numSamplesOk; o++ )
            {

                this->m_AdvancedTransform->GetJacobianOfSpatialHessian( fixedPoints[o], spatialHessian, jacobianOfSpatialHessian, nonZeroJacobianIndices );
                
                FixedArray< InternalMatrixType, ReducedFixedImageDimension > A;
                for( unsigned int k = 0; k < ReducedFixedImageDimension; k++ )
                {
                    A[ k ] = spatialHessian[ k ].GetVnlMatrix();
                }
                
                for( unsigned int k = 0; k < ReducedFixedImageDimension; k++ )
                {
                    measure += vnl_math_sqr(spatialHessian[ k ].GetVnlMatrix().frobenius_norm() ) / numSamplesOk;
                }
                
                if( !this->m_TransformIsBSpline )
                {
                    if(!this->m_SubTransformIsBSpline)
                    {
                        for( unsigned int mu = 0; mu < nonZeroJacobianIndices.size(); ++mu )
                        {
                            for( unsigned int k = 0; k < ReducedFixedImageDimension; ++k )
                            {
                                const InternalMatrixType & B = jacobianOfSpatialHessian[ mu ][ k ].GetVnlMatrix();
                            
                                RealType matrixProduct = 0.0;
                                typename InternalMatrixType::const_iterator itA    = A[ k ].begin();
                                typename InternalMatrixType::const_iterator itB    = B.begin();
                                typename InternalMatrixType::const_iterator itAend = A[ k ].end();
                                while( itA != itAend )
                                {
                                    matrixProduct += ( *itA ) * ( *itB );
                                    ++itA;
                                    ++itB;
                                }
                            
                                derivative[ nonZeroJacobianIndices[ mu ] ] += 2.0 * matrixProduct /numSamplesOk;
                            }
                        }
                    }
                    else
                    {
                        unsigned int numParPerDim
                        = nonZeroJacobianIndices.size() / ReducedFixedImageDimension;
                        
                        for( unsigned int mu = 0; mu < numParPerDim; ++mu )
                        {
                            for( unsigned int k = 0; k < ReducedFixedImageDimension; ++k )
                            {
                                
                                const InternalMatrixType & B
                                = jacobianOfSpatialHessian[ mu + numParPerDim * k ][ k ].GetVnlMatrix();
                                
                                RealType matrixElementProduct = 0.0;
                                typename InternalMatrixType::const_iterator itA    = A[ k ].begin();
                                typename InternalMatrixType::const_iterator itB    = B.begin();
                                typename InternalMatrixType::const_iterator itAend = A[ k ].end();
                                while( itA != itAend )
                                {
                                    matrixElementProduct += ( *itA ) * ( *itB );
                                    ++itA;
                                    ++itB;
                                }
                                
                                derivative[ nonZeroJacobianIndices[ mu + numParPerDim * k ] ]
                                += 2.0 * matrixElementProduct /numSamplesOk;
                            }
                        }
                    }
                }
                else
                {

                    unsigned int numParPerDim
                    = nonZeroJacobianIndices.size() / FixedImageDimension;

                    for( unsigned int mu = 0; mu < numParPerDim; ++mu )
                    {
                        for( unsigned int k = 0; k < ReducedFixedImageDimension; ++k )
                        {

                            const InternalMatrixType & B
                            = jacobianOfSpatialHessian[ mu + numParPerDim * k ][ k ].GetVnlMatrix();
                            
                            RealType matrixElementProduct = 0.0;
                            typename InternalMatrixType::const_iterator itA    = A[ k ].begin();
                            typename InternalMatrixType::const_iterator itB    = B.begin();
                            typename InternalMatrixType::const_iterator itAend = A[ k ].end();
                            while( itA != itAend )
                            {
                                matrixElementProduct += ( *itA ) * ( *itB );
                                ++itA;
                                ++itB;
                            }
                            
                            derivative[ nonZeroJacobianIndices[ mu + numParPerDim * k ] ]
                            += 2.0 * matrixElementProduct /numSamplesOk;
                        }
                    }
                }
            }
        }
    }
    
    this->CheckNumberOfSamples( sampleContainer->Size(), this->m_NumberOfPixelsCounted );
    if( this->m_SubtractMean )
    {
        if( !this->m_TransformIsStackTransform )
        {
            const unsigned int lastDimGridSize              = this->m_GridSize[ ReducedFixedImageDimension ];
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
            const unsigned int numParametersPerLastDimension = this->GetNumberOfParameters() / this->GetFixedImage()->GetLargestPossibleRegion().GetSize( ReducedFixedImageDimension );
            DerivativeType     mean( numParametersPerLastDimension );
            mean.Fill( 0.0 );
            
            for( unsigned int t = 0; t < this->GetFixedImage()->GetLargestPossibleRegion().GetSize( ReducedFixedImageDimension ); ++t )
            {
                const unsigned int startc = numParametersPerLastDimension * t;
                for( unsigned int c = startc; c < startc + numParametersPerLastDimension; ++c )
                {
                    const unsigned int index = c % numParametersPerLastDimension;
                    mean[ index ] += derivative[ c ];
                }
            }
            mean /= static_cast< double >( this->GetFixedImage()->GetLargestPossibleRegion().GetSize( ReducedFixedImageDimension ) );
            
            for( unsigned int t = 0; t < this->GetFixedImage()->GetLargestPossibleRegion().GetSize( ReducedFixedImageDimension ); ++t )
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
    
    measure    /= static_cast< RealType >( this->m_NumberOfPixelsCounted );
    derivative /= static_cast< RealType >( this->m_NumberOfPixelsCounted );

    value = static_cast< MeasureType >( measure );

} // end GetValueAndDerivativeSingleThreaded()


/**
 * ******************* GetValueAndDerivative *******************
 */

template< class TFixedImage, class TScalarType >
void
StackTransformBendingEnergyPenaltyTerm< TFixedImage, TScalarType >
::GetValueAndDerivative(
  const ParametersType & parameters,
  MeasureType & value, DerivativeType & derivative ) const
{
  if( !this->m_UseMultiThread )
  {
    return this->GetValueAndDerivativeSingleThreaded(
      parameters, value, derivative );
  }

  this->BeforeThreadedGetValueAndDerivative( parameters );

    if(this->m_SampleLastDimensionRandomly)
    {
        this->m_RandomList.resize(this->m_NumSamplesLastDimension);
        this->SampleRandom(this->GetFixedImage()->GetLargestPossibleRegion().GetSize( ReducedFixedImageDimension ),this->m_NumSamplesLastDimension,this->m_RandomList);
    }
    else
    {
        this->m_NumSamplesLastDimension = this->GetFixedImage()->GetLargestPossibleRegion().GetSize(ReducedFixedImageDimension);
        this->m_RandomList.resize(this->m_NumSamplesLastDimension);
        for( unsigned int d = 0; d < this->m_NumSamplesLastDimension; d++ )
        {
            this->m_RandomList[d]=d;
        }
        
    }

  this->LaunchGetValueAndDerivativeThreaderCallback();

  this->AfterThreadedGetValueAndDerivative( value, derivative );

} // end GetValueAndDerivative()


/**
 * ******************* ThreadedGetValueAndDerivative *******************
 */

template< class TFixedImage, class TScalarType >
void
StackTransformBendingEnergyPenaltyTerm< TFixedImage, TScalarType >
::ThreadedGetValueAndDerivative( ThreadIdType threadId )
{
    
    SpatialHessianType           spatialHessian;
    JacobianOfSpatialHessianType jacobianOfSpatialHessian;
    NonZeroJacobianIndicesType   nonZeroJacobianIndices;
    const NumberOfParametersType numberOfNonZeroJacobianIndices
    = this->m_AdvancedTransform->GetNumberOfNonZeroJacobianIndices();
    jacobianOfSpatialHessian.resize( numberOfNonZeroJacobianIndices );
    nonZeroJacobianIndices.resize( numberOfNonZeroJacobianIndices );
    
    if( !this->m_AdvancedTransform->GetHasNonZeroSpatialHessian()
       && !this->m_AdvancedTransform->GetHasNonZeroJacobianOfSpatialHessian() )
    {
        return;
    }
    
    DerivativeType & derivative = this->m_GetValueAndDerivativePerThreadVariables[ threadId ].st_Derivative;

    ImageSampleContainerPointer sampleContainer = this->GetImageSampler()->GetOutput();

    const unsigned long         sampleContainerSize = sampleContainer->Size();
    
    const unsigned long nrOfSamplesPerThreads
    = static_cast< unsigned long >( vcl_ceil( static_cast< double >( sampleContainerSize ) / static_cast< double >( this->m_NumberOfThreads ) ) );
    
    unsigned long pos_begin = nrOfSamplesPerThreads * threadId;
    unsigned long pos_end   = nrOfSamplesPerThreads * ( threadId + 1 );
    pos_begin = ( pos_begin > sampleContainerSize ) ? sampleContainerSize : pos_begin;
    pos_end   = ( pos_end > sampleContainerSize ) ? sampleContainerSize : pos_end;

    typename ImageSampleContainerType::ConstIterator fiter;
    typename ImageSampleContainerType::ConstIterator fbegin = sampleContainer->Begin();
    typename ImageSampleContainerType::ConstIterator fend   = sampleContainer->Begin();
    fbegin                                                 += (int)pos_begin;
    fend                                                   += (int)pos_end;

    unsigned long numberOfPixelsCounted = 0;
    MeasureType   measure               = NumericTraits< MeasureType >::Zero;

    for( fiter = fbegin; fiter != fend; ++fiter )
    {

        const FixedImagePointType & fixedPoint = ( *fiter ).Value().m_ImageCoordinates;
        
        FixedImageContinuousIndexType voxelCoord;
        
        this->GetFixedImage()->TransformPhysicalPointToContinuousIndex( fixedPoint, voxelCoord );
        
        std::vector< FixedImagePointType >  fixedPoints(this->m_NumSamplesLastDimension);
        
        unsigned int       numSamplesOk            = 0;

        for( unsigned int s = 0; s < this->m_NumSamplesLastDimension; s++ )
        {
            RealType             movingImageValueTemp;
            MovingImagePointType mappedPoint;
            FixedImagePointType tempPoint;

            voxelCoord[ ReducedFixedImageDimension ] = this->m_RandomList[ s ];
            
            this->GetFixedImage()->TransformContinuousIndexToPhysicalPoint( voxelCoord, tempPoint );
            
            bool sampleOk = this->TransformPoint( tempPoint, mappedPoint );
            
            if( sampleOk )
            {
                sampleOk = this->IsInsideMovingMask( mappedPoint );
            }
            
            if( sampleOk )
            {
                fixedPoints[numSamplesOk] = tempPoint;
                numSamplesOk++;
                
            }
            
        }

        if (numSamplesOk > 0)
        {
            numberOfPixelsCounted++;
            fixedPoints.resize(numSamplesOk);

            for( unsigned int o = 0; o < numSamplesOk; o++ )
            {
                this->m_AdvancedTransform->GetJacobianOfSpatialHessian( fixedPoints[o], spatialHessian, jacobianOfSpatialHessian, nonZeroJacobianIndices );
                
                FixedArray< InternalMatrixType, ReducedFixedImageDimension > A;
                for( unsigned int k = 0; k < ReducedFixedImageDimension; k++ )
                {
                    A[ k ] = spatialHessian[ k ].GetVnlMatrix();
                }
                
                for( unsigned int k = 0; k < ReducedFixedImageDimension; k++ )
                {
                    measure += vnl_math_sqr(spatialHessian[ k ].GetVnlMatrix().frobenius_norm() ) / numSamplesOk;
                }
 
                if( !this->m_TransformIsBSpline )
                {
                    if(!this->m_SubTransformIsBSpline)
                    {
                        for( unsigned int mu = 0; mu < nonZeroJacobianIndices.size(); ++mu )
                        {
                            for( unsigned int k = 0; k < ReducedFixedImageDimension; ++k )
                            {
                                const InternalMatrixType & B = jacobianOfSpatialHessian[ mu ][ k ].GetVnlMatrix();
                                
                                RealType matrixProduct = 0.0;
                                typename InternalMatrixType::const_iterator itA    = A[ k ].begin();
                                typename InternalMatrixType::const_iterator itB    = B.begin();
                                typename InternalMatrixType::const_iterator itAend = A[ k ].end();
                                while( itA != itAend )
                                {
                                    matrixProduct += ( *itA ) * ( *itB );
                                    ++itA;
                                    ++itB;
                                }
                                
                                derivative[ nonZeroJacobianIndices[ mu ] ] += 2.0 * matrixProduct /numSamplesOk;
                            }
                        }
                    }
                    else
                    {
                        unsigned int numParPerDim
                        = nonZeroJacobianIndices.size() / ReducedFixedImageDimension;
                        for( unsigned int mu = 0; mu < numParPerDim; ++mu )
                        {
                            for( unsigned int k = 0; k < ReducedFixedImageDimension; ++k )
                            {
                                const InternalMatrixType & B
                                = jacobianOfSpatialHessian[ mu + numParPerDim * k ][ k ].GetVnlMatrix();
                                
                                RealType matrixElementProduct = 0.0;
                                typename InternalMatrixType::const_iterator itA    = A[ k ].begin();
                                typename InternalMatrixType::const_iterator itB    = B.begin();
                                typename InternalMatrixType::const_iterator itAend = A[ k ].end();

                                while( itA != itAend )
                                {
                                    matrixElementProduct += ( *itA ) * ( *itB );
                                    ++itA;
                                    ++itB;
                                }
                                derivative[ nonZeroJacobianIndices[ mu + numParPerDim * k ] ] += 2.0 * matrixElementProduct /numSamplesOk;

                            }
                        }
                    }
                }
                else
                {
                    
                    unsigned int numParPerDim
                    = nonZeroJacobianIndices.size() / FixedImageDimension;
                    
                    for( unsigned int mu = 0; mu < numParPerDim; ++mu )
                    {
                        for( unsigned int k = 0; k < ReducedFixedImageDimension; ++k )
                        {
                            
                            const InternalMatrixType & B
                            = jacobianOfSpatialHessian[ mu + numParPerDim * k ][ k ].GetVnlMatrix();
                            
                            RealType matrixElementProduct = 0.0;
                            typename InternalMatrixType::const_iterator itA    = A[ k ].begin();
                            typename InternalMatrixType::const_iterator itB    = B.begin();
                            typename InternalMatrixType::const_iterator itAend = A[ k ].end();
                            while( itA != itAend )
                            {
                                matrixElementProduct += ( *itA ) * ( *itB );
                                ++itA;
                                ++itB;
                            }
                            
                            derivative[ nonZeroJacobianIndices[ mu + numParPerDim * k ] ]
                            += 2.0 * matrixElementProduct /numSamplesOk;
                        }
                    }
                }
            }
        }
    }

    this->m_GetValueAndDerivativePerThreadVariables[ threadId ].st_NumberOfPixelsCounted = numberOfPixelsCounted;
    this->m_GetValueAndDerivativePerThreadVariables[ threadId ].st_Value                 = measure;

} // end ThreadedGetValueAndDerivative()


/**
 * ******************* AfterThreadedGetValueAndDerivative *******************
 */

template< class TFixedImage, class TScalarType >
void
StackTransformBendingEnergyPenaltyTerm< TFixedImage, TScalarType >
::AfterThreadedGetValueAndDerivative(
  MeasureType & value, DerivativeType & derivative ) const
{
    this->m_NumberOfPixelsCounted = 0;
    for( ThreadIdType i = 0; i < this->m_NumberOfThreads; ++i )
    {
        this->m_NumberOfPixelsCounted += this->m_GetValueAndDerivativePerThreadVariables[ i ].st_NumberOfPixelsCounted;
        
        /** Reset this variable for the next iteration. */
        this->m_GetValueAndDerivativePerThreadVariables[ i ].st_NumberOfPixelsCounted = 0;
    }

    ImageSampleContainerPointer sampleContainer = this->GetImageSampler()->GetOutput();
    this->CheckNumberOfSamples( sampleContainer->Size(), this->m_NumberOfPixelsCounted );
    
    value = NumericTraits< MeasureType >::Zero;
    for( ThreadIdType i = 0; i < this->m_NumberOfThreads; ++i )
    {
        value += this->m_GetValueAndDerivativePerThreadVariables[ i ].st_Value;
        
        /** Reset this variable for the next iteration. IS THIS REALLY NESSACARY???*/
        this->m_GetValueAndDerivativePerThreadVariables[ i ].st_Value = NumericTraits< MeasureType >::Zero;
    }
    value /= static_cast< RealType >( this->m_NumberOfPixelsCounted );

    derivative = this->m_GetValueAndDerivativePerThreadVariables[ 0 ].st_Derivative;
    this->m_GetValueAndDerivativePerThreadVariables[ 0 ].st_Derivative.Fill(0.0);
    for( ThreadIdType i = 1; i < this->m_NumberOfThreads; ++i )
    {
        derivative += this->m_GetValueAndDerivativePerThreadVariables[ i ].st_Derivative;
        this->m_GetValueAndDerivativePerThreadVariables[ i ].st_Derivative.Fill(0.0);

    }
    derivative /= static_cast< DerivativeValueType >( this->m_NumberOfPixelsCounted );
    
    if( this->m_SubtractMean )
    {
        if( !this->m_TransformIsStackTransform )
        {
            const unsigned int lastDimGridSize              = this->m_GridSize[ ReducedFixedImageDimension ];
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
            const unsigned int numParametersPerLastDimension = this->GetNumberOfParameters() / this->GetFixedImage()->GetLargestPossibleRegion().GetSize( ReducedFixedImageDimension );
            DerivativeType     mean( numParametersPerLastDimension );
            mean.Fill( 0.0 );
            
            for( unsigned int t = 0; t < this->GetFixedImage()->GetLargestPossibleRegion().GetSize( ReducedFixedImageDimension ); ++t )
            {
                const unsigned int startc = numParametersPerLastDimension * t;
                for( unsigned int c = startc; c < startc + numParametersPerLastDimension; ++c )
                {
                    const unsigned int index = c % numParametersPerLastDimension;
                    mean[ index ] += derivative[ c ];
                }
            }
            mean /= static_cast< double >( this->GetFixedImage()->GetLargestPossibleRegion().GetSize( ReducedFixedImageDimension ) );
            
            for( unsigned int t = 0; t < this->GetFixedImage()->GetLargestPossibleRegion().GetSize( ReducedFixedImageDimension ); ++t )
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

} // end AfterThreadedGetValueAndDerivative()

/**
 * ******************* SampleRandom *******************
 */

template< class TFixedImage, class TScalarType >
void
StackTransformBendingEnergyPenaltyTerm< TFixedImage, TScalarType >
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
    
} // end namespace itk



#endif // #ifndef __itkStackTransformBendingEnergyPenaltyTerm_hxx
