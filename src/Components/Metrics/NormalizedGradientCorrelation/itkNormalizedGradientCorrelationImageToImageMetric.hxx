
#ifndef __itkNormalizedGradientCorrelationImageToImageMetric_hxx
#define __itkNormalizedGradientCorrelationImageToImageMetric_hxx

#include "itkNormalizedGradientCorrelationImageToImageMetric.h"
#include "itkImageRegionConstIteratorWithIndex.h"
#include "itkNumericTraits.h"

#include "itkImageFileWriter.h"

#include <iostream>
#include <iomanip>
#include <stdio.h>

#include "itkSimpleFilterWatcher.h"
namespace itk
{

/**
 * Constructor
 */
template <class TFixedImage, class TMovingImage>
NormalizedGradientCorrelationImageToImageMetric<TFixedImage,TMovingImage>
::NormalizedGradientCorrelationImageToImageMetric()
{

  m_CastFixedImageFilter = CastFixedImageFilterType::New();
  m_CastMovedImageFilter = CastMovedImageFilterType::New();
  m_CombinationTransform = CombinationTransformType::New();
  m_TransformMovingImageFilter = TransformMovingImageFilterType::New();

  this->m_DerivativeDelta = 0.001;

  unsigned iDimension = 0;

  for (iDimension=0; iDimension<MovedImageDimension; iDimension++)
    {
    m_MeanFixedGradient[iDimension] = 0;
    m_MeanMovedGradient[iDimension] = 0;
    }

}


/**
 * Initialize
 */
template <class TFixedImage, class TMovingImage>
void
NormalizedGradientCorrelationImageToImageMetric<TFixedImage,TMovingImage>
::Initialize(void) throw ( ExceptionObject )
{
  unsigned int iFilter;  // Index of Sobel filters for each dimension

  /** Initialise the base class */

  Superclass::Initialize();

  typedef typename FixedImageType::SizeType SizeType;
  SizeType size = this->m_FixedImage->GetLargestPossibleRegion().GetSize();

  /** Compute the gradient of the fixed images */

  m_CastFixedImageFilter->SetInput( this->m_FixedImage );
  m_CastFixedImageFilter->Update();

  for (iFilter=0; iFilter<FixedImageDimension; iFilter++)
    {
    m_FixedSobelOperators[iFilter].SetDirection( iFilter );

    m_FixedSobelOperators[iFilter].CreateDirectional();

    m_FixedSobelFilters[iFilter] = FixedSobelFilter::New();

    m_FixedSobelFilters[iFilter]->OverrideBoundaryCondition( &m_FixedBoundCond );

    m_FixedSobelFilters[iFilter]->SetOperator( m_FixedSobelOperators[iFilter] );

    m_FixedSobelFilters[iFilter]->SetInput( m_CastFixedImageFilter->GetOutput() );

    m_FixedSobelFilters[iFilter]->UpdateLargestPossibleRegion();
    }

  this->ComputeMeanFixedGradient();

  /* resampling for 3D->2D */

  m_TransformMovingImageFilter->SetTransform( dynamic_cast<CombinationTransformType *>(
		  		dynamic_cast<RayCastInterpolatorType *>(
			  		const_cast<  InterpolatorType *>(
				  	(this->GetInterpolator())
					)
				)->GetTransform()) );
  m_TransformMovingImageFilter->SetInterpolator( this->m_Interpolator );
  m_TransformMovingImageFilter->SetInput( this->m_MovingImage );

  m_TransformMovingImageFilter->SetDefaultPixelValue( 0 );

  m_TransformMovingImageFilter->SetSize( this->m_FixedImage->GetLargestPossibleRegion().GetSize() );
  m_TransformMovingImageFilter->SetOutputOrigin( this->m_FixedImage->GetOrigin() );
  m_TransformMovingImageFilter->SetOutputSpacing( this->m_FixedImage->GetSpacing() );
  m_TransformMovingImageFilter->SetOutputDirection( this->m_FixedImage->GetDirection() );
  m_TransformMovingImageFilter->Update();

  m_CastMovedImageFilter->SetInput( m_TransformMovingImageFilter->GetOutput() );

  for (iFilter=0; iFilter<MovedImageDimension; iFilter++)
    {
    m_MovedSobelOperators[iFilter].SetDirection( iFilter );
    m_MovedSobelOperators[iFilter].CreateDirectional();

    m_MovedSobelFilters[iFilter] = MovedSobelFilter::New();

    m_MovedSobelFilters[iFilter]->OverrideBoundaryCondition( &m_MovedBoundCond );
    m_MovedSobelFilters[iFilter]->SetOperator( m_MovedSobelOperators[iFilter] );

    m_MovedSobelFilters[iFilter]->SetInput( m_CastMovedImageFilter->GetOutput() );

    m_MovedSobelFilters[iFilter]->UpdateLargestPossibleRegion();

    }

}


/**
 * PrintSelf
 */
template <class TFixedImage, class TMovingImage>
void
NormalizedGradientCorrelationImageToImageMetric<TFixedImage,TMovingImage>
::PrintSelf(std::ostream& os, Indent indent) const
{
  Superclass::PrintSelf( os, indent );
  os << indent << "DerivativeDelta: " << this->m_DerivativeDelta << std::endl;
}

/**
 * Get the mean of the fixed gradients
 */
template <class TFixedImage, class TMovingImage>
void
NormalizedGradientCorrelationImageToImageMetric<TFixedImage,TMovingImage>
::ComputeMeanFixedGradient( void ) const
{


  typename FixedGradientImageType::IndexType currentIndex;
  typename FixedGradientImageType::PointType point;

    for (int iDimension=0; iDimension<FixedImageDimension; iDimension++)
      {

 	m_FixedSobelFilters[iDimension]->UpdateLargestPossibleRegion();

      }

   typedef  itk::ImageRegionConstIteratorWithIndex< FixedGradientImageType >
     FixedIteratorType;

    FixedIteratorType fixedIteratorx( m_FixedSobelFilters[0]->GetOutput(),
                                     this->GetFixedImageRegion() );
    FixedIteratorType fixedIteratory( m_FixedSobelFilters[1]->GetOutput(),
                                     this->GetFixedImageRegion() );

    fixedIteratorx.GoToBegin();
    fixedIteratory.GoToBegin();

    bool sampleOK = false;
    FixedGradientPixelType fixedGradient[FixedImageDimension];
		for (int i = 0; i < FixedImageDimension; i++)
			fixedGradient[i] = 0.0;
    unsigned long nPixels = 0;

    if ( this->m_FixedImageMask.IsNull() )
		sampleOK = true;

    while ( ! fixedIteratorx.IsAtEnd() ) {

      /** Get current index */

      currentIndex = fixedIteratorx.GetIndex();
      this->m_FixedImage->TransformIndexToPhysicalPoint( currentIndex, point );

      /** if fixedMask is given */

      if ( !this->m_FixedImageMask.IsNull() ){

        if ( this->m_FixedImageMask->IsInside( point ) ) // sample is good
	  sampleOK = true;
	else // sample no good
	  sampleOK = false;
      }

      if(sampleOK){ //go on

      // Get the moving and fixed image gradients

      	fixedGradient[0] += fixedIteratorx.Get();
      	fixedGradient[1] += fixedIteratory.Get();
	nPixels++;

      } // end if sampleOK

      	++fixedIteratorx;
      	++fixedIteratory;

    } // end while

  m_MeanFixedGradient[0] = fixedGradient[0]/nPixels;
  m_MeanFixedGradient[1] = fixedGradient[1]/nPixels;
}

/**
 * Get the mean of the moved gradients
 */
template <class TFixedImage, class TMovingImage>
void
NormalizedGradientCorrelationImageToImageMetric<TFixedImage,TMovingImage>
::ComputeMeanMovedGradient( void ) const
{

  typename MovedGradientImageType::IndexType currentIndex;
  typename MovedGradientImageType::PointType point;

    for (int iDimension=0; iDimension<MovedImageDimension; iDimension++)
      {

 	m_MovedSobelFilters[iDimension]->UpdateLargestPossibleRegion();

      }

   typedef  itk::ImageRegionConstIteratorWithIndex< MovedGradientImageType >
     MovedIteratorType;

    MovedIteratorType movedIteratorx( m_MovedSobelFilters[0]->GetOutput(),
                                     this->GetFixedImageRegion() );
    MovedIteratorType movedIteratory( m_MovedSobelFilters[1]->GetOutput(),
                                     this->GetFixedImageRegion() );

    movedIteratorx.GoToBegin();
    movedIteratory.GoToBegin();

    bool sampleOK = false;

    if ( this->m_FixedImageMask.IsNull() )
		sampleOK = true;

    MovedGradientPixelType movedGradient[MovedImageDimension];
		for (int i = 0; i < MovedImageDimension; i++)
			movedGradient[i] = 0.0;

    unsigned long nPixels = 0;

    while ( ! movedIteratorx.IsAtEnd() ) {

      /** Get current index */

      currentIndex = movedIteratorx.GetIndex();
      this->m_FixedImage->TransformIndexToPhysicalPoint( currentIndex, point );

      /** if fixedMask is given */

      if ( !this->m_FixedImageMask.IsNull() ){


        if ( this->m_FixedImageMask->IsInside( point ) ) // sample is good
	  sampleOK = true;
	else // sample no good
	  sampleOK = false;
      }

      if(sampleOK){ //go on

      // Get the moving and fixed image gradients

      	movedGradient[0] += movedIteratorx.Get();
      	movedGradient[1] += movedIteratory.Get();
	nPixels++;

      } // end if sampleOK

      	++movedIteratorx;
      	++movedIteratory;

    } // end while

  m_MeanMovedGradient[0] = movedGradient[0]/nPixels;
  m_MeanMovedGradient[1] = movedGradient[1]/nPixels;
}

/**
 * Get the value of the similarity measure
 */
template <class TFixedImage, class TMovingImage>
typename NormalizedGradientCorrelationImageToImageMetric<TFixedImage,TMovingImage>::MeasureType
NormalizedGradientCorrelationImageToImageMetric<TFixedImage,TMovingImage>
::ComputeMeasure( const TransformParametersType & parameters ) const
{

  this->SetTransformParameters( parameters );

  m_TransformMovingImageFilter->Modified();
  m_TransformMovingImageFilter->UpdateLargestPossibleRegion();

  /* DEBUG */
  //std::cout<<"debugging! current parameters: "<<parameters<<std::endl;
  typedef itk::ImageFileWriter<TransformedMovingImageType> WriterType;
  typename WriterType::Pointer writer = WriterType::New();
  writer->SetFileName("debug.mhd");
  writer->SetInput(m_TransformMovingImageFilter->GetOutput());
  writer->Update();


  typename FixedImageType::IndexType currentIndex;
  typename FixedImageType::PointType point;

  MeasureType measure = NumericTraits< MeasureType >::Zero;

  MovedGradientPixelType NmovedGradient[FixedImageDimension];
  FixedGradientPixelType NfixedGradient[FixedImageDimension];

  MeasureType NGcrosscorrelation = NumericTraits< MeasureType >::Zero;
  MeasureType NGautocorrelationfixed = NumericTraits< MeasureType >::Zero;
  MeasureType NGautocorrelationmoving = NumericTraits< MeasureType >::Zero;

  /** Make sure all is updated */
    for (int iDimension=0; iDimension<FixedImageDimension; iDimension++)
      {

 	m_FixedSobelFilters[iDimension]->UpdateLargestPossibleRegion();
    	m_MovedSobelFilters[iDimension]->UpdateLargestPossibleRegion();

      }

   typedef  itk::ImageRegionConstIteratorWithIndex< FixedGradientImageType >
     FixedIteratorType;

    FixedIteratorType fixedIteratorx( m_FixedSobelFilters[0]->GetOutput(),
                                     this->GetFixedImageRegion() );
    FixedIteratorType fixedIteratory( m_FixedSobelFilters[1]->GetOutput(),
                                     this->GetFixedImageRegion() );

    fixedIteratorx.GoToBegin();
    fixedIteratory.GoToBegin();

    typedef  itk::ImageRegionConstIteratorWithIndex< MovedGradientImageType >
      MovedIteratorType;

    MovedIteratorType movedIteratorx( m_MovedSobelFilters[0]->GetOutput(),
                                     this->GetFixedImageRegion() );
    MovedIteratorType movedIteratory( m_MovedSobelFilters[1]->GetOutput(),
                                     this->GetFixedImageRegion() );

    movedIteratorx.GoToBegin();
    movedIteratory.GoToBegin();

    this->m_NumberOfPixelsCounted = 0;

    bool sampleOK = false;

    if ( this->m_FixedImageMask.IsNull() )
		sampleOK = true;

    while ( ! fixedIteratorx.IsAtEnd() ) {

      /** Get current index */

      currentIndex = fixedIteratorx.GetIndex();
      this->m_FixedImage->TransformIndexToPhysicalPoint( currentIndex, point );

      /** if fixedMask is given */

      if ( !this->m_FixedImageMask.IsNull() ){


        if ( this->m_FixedImageMask->IsInside( point ) ) // sample is good
	  sampleOK = true;
	else // sample no good
	  sampleOK = false;
      }

      if(sampleOK){ //go on

      // Get the moving and fixed image gradients

      	NmovedGradient[0] = movedIteratorx.Get() - m_MeanMovedGradient[0];
      	NfixedGradient[0] = fixedIteratorx.Get() - m_MeanFixedGradient[0];
      	NmovedGradient[1] = movedIteratory.Get() - m_MeanMovedGradient[1];
      	NfixedGradient[1] = fixedIteratory.Get() - m_MeanFixedGradient[1];
	NGcrosscorrelation += NmovedGradient[0] * NfixedGradient[0] + NmovedGradient[1] * NfixedGradient[1];
	NGautocorrelationmoving += NmovedGradient[0] * NmovedGradient[0] + NmovedGradient[1] * NmovedGradient[1];
	NGautocorrelationfixed += NfixedGradient[0] * NfixedGradient[0] + NfixedGradient[1] * NfixedGradient[1];

      } // end if sampleOK

      	++fixedIteratorx;
      	++fixedIteratory;
      	++movedIteratorx;
      	++movedIteratory;

    } // end while

  measure = -1 * (NGcrosscorrelation/(sqrt(NGautocorrelationfixed) * sqrt(NGautocorrelationmoving)));
  return measure;
}


/**
 * Get the value of the similarity measure
 */
template <class TFixedImage, class TMovingImage>
typename NormalizedGradientCorrelationImageToImageMetric<TFixedImage,TMovingImage>::MeasureType
NormalizedGradientCorrelationImageToImageMetric<TFixedImage,TMovingImage>
::GetValue( const TransformParametersType & parameters ) const
{
  unsigned int iFilter;                        // Index of Sobel filters for each dimension

  this->SetTransformParameters( parameters );

  /** Update the imageSampler and get a handle to the sample container. */

  m_TransformMovingImageFilter->Modified();
  m_TransformMovingImageFilter->UpdateLargestPossibleRegion();

  // Update the gradient images

  for (iFilter=0; iFilter<MovedImageDimension; iFilter++)
    {
    m_MovedSobelFilters[iFilter]->UpdateLargestPossibleRegion();
    }

  this->ComputeMeanMovedGradient();
  MeasureType currentMeasure = NumericTraits< MeasureType >::Zero;
  currentMeasure = this->ComputeMeasure( parameters );

  return currentMeasure;
}

/**
 * Set the parameters that define a unique transform
 */
template <class TFixedImage, class TMovingImage>
void
NormalizedGradientCorrelationImageToImageMetric<TFixedImage,TMovingImage>
::SetTransformParameters( const TransformParametersType & parameters ) const
{
  if( !this->m_Transform )
    {
    itkExceptionMacro(<<"Transform has not been assigned");
    }
  this->m_Transform->SetParameters( parameters );

}

/**
 * Get the Derivative Measure
 */
template < class TFixedImage, class TMovingImage>
void
NormalizedGradientCorrelationImageToImageMetric<TFixedImage,TMovingImage>
::GetDerivative( const TransformParametersType & parameters,
                 DerivativeType & derivative           ) const
{

  TransformParametersType testPoint;
  testPoint = parameters;

  const unsigned int numberOfParameters = this->GetNumberOfParameters();
  derivative = DerivativeType( numberOfParameters );

  for( unsigned int i=0; i<numberOfParameters; i++)
    {
    testPoint[i] -= this->m_DerivativeDelta / sqrt(m_Scales[i]);
    const MeasureType valuep0 = this->GetValue( testPoint );
    testPoint[i] += 2* this->m_DerivativeDelta / sqrt(m_Scales[i]);
    const MeasureType valuep1 = this->GetValue( testPoint );
    derivative[i] = (valuep1 - valuep0 ) / ( 2 * this->m_DerivativeDelta / sqrt(m_Scales[i]) );
    testPoint[i] = parameters[i];
    }

}


/**
 * Get both the match Measure and theDerivative Measure
 */
template <class TFixedImage, class TMovingImage>
void
NormalizedGradientCorrelationImageToImageMetric<TFixedImage,TMovingImage>
::GetValueAndDerivative(const TransformParametersType & parameters,
                        MeasureType & Value, DerivativeType  & Derivative) const
{
  Value      = this->GetValue( parameters );
  this->GetDerivative( parameters, Derivative );
}

} // end namespace itk


#endif


