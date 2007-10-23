#ifndef _itkKappaStatisticImageToImageMetric2_txx
#define _itkKappaStatisticImageToImageMetric2_txx

#include "itkKappaStatisticImageToImageMetric2.h"
//#include "itkImageRegionConstIteratorWithIndex.h"
//#include "itkImageRegionIteratorWithIndex.h"

namespace itk
{

/**
 * ******************* Constructor *******************
 */

template <class TFixedImage, class TMovingImage> 
KappaStatisticImageToImageMetric2<TFixedImage,TMovingImage>
::KappaStatisticImageToImageMetric2()
{
  itkDebugMacro( "Constructor" );

  this->SetUseImageSampler( true );
  this->SetComputeGradient( true );

  this->m_ForegroundValue = 1.0;
  this->m_Complement = false;
} // end Constructor()


/**
 * ******************* PrintSelf *******************
 */

template <class TFixedImage, class TMovingImage> 
void
KappaStatisticImageToImageMetric2<TFixedImage,TMovingImage>
::PrintSelf(std::ostream& os, Indent indent) const
{
  Superclass::PrintSelf( os, indent );
  os << indent << "Complement: "      << ( this->m_Complement ? "On" : "Off" ) << std::endl; 
  os << indent << "ForegroundValue: " << this->m_ForegroundValue << std::endl;
} // end PrintSelf()


/**
 * ******************* GetValue *******************
 */

template <class TFixedImage, class TMovingImage> 
typename KappaStatisticImageToImageMetric2<TFixedImage,TMovingImage>::MeasureType
KappaStatisticImageToImageMetric2<TFixedImage,TMovingImage>
::GetValue( const TransformParametersType & parameters ) const
{
  itkDebugMacro("GetValue( " << parameters << " ) ");

  /** Make sure the transform parameters are up to date. */
	this->SetTransformParameters( parameters );

  this->m_NumberOfPixelsCounted = 0;

  /** Update the imageSampler and get a handle to the sample container. */
  this->GetImageSampler()->Update();
  ImageSampleContainerPointer sampleContainer = this->GetImageSampler()->GetOutput();

  /** Create iterator over the sample container. */
  typename ImageSampleContainerType::ConstIterator iter;
  typename ImageSampleContainerType::ConstIterator begin = sampleContainer->Begin();
  typename ImageSampleContainerType::ConstIterator end = sampleContainer->End();

  /** Create variables to store intermediate results.
   * 'measure' is the value of the metric.
   * 'fixedForegroundArea' is the total area of the foreground region
   *    in the fixed image.
   * 'movingForegroundArea' is the foreground area in the moving image
   *    in the area of overlap under the current transformation.
   * 'intersection' is the area of foreground intersection between the
   *    fixed and moving image.
   */
  MeasureType measure              = NumericTraits< MeasureType >::Zero;
  MeasureType fixedForegroundArea  = NumericTraits< MeasureType >::Zero;
  MeasureType movingForegroundArea = NumericTraits< MeasureType >::Zero;
  MeasureType intersection         = NumericTraits< MeasureType >::Zero;
	InputPointType  inputPoint;
	OutputPointType transformedPoint;

  /** Loop over the fixed image samples to calculate the kappa statistic. */
  for ( iter = begin; iter != end; ++iter )
  {
    /** Get the current inputpoint. */
    inputPoint = (*iter).Value().m_ImageCoordinates;

		/** Transform the inputpoint to get the transformed point. */
	  transformedPoint = this->m_Transform->TransformPoint( inputPoint );

    /** Inside the moving image mask? */
		if ( this->m_MovingImageMask && !this->m_MovingImageMask->IsInside( transformedPoint ) )
		{
			continue;
		}

    /** In this if-statement the actual calculation of kappa statistic is done. */
    if ( this->m_Interpolator->IsInsideBuffer( transformedPoint ) )
    {
      /** Get the fixedValue = f(x) and the movingValue = m(x+u(x)). */
      const RealType movingValue  = this->m_Interpolator->Evaluate( transformedPoint );
      const RealType & fixedValue = (*iter).Value().m_ImageValue;

      /** Update the fixed foreground value. */
      if ( fixedValue == this->m_ForegroundValue ) fixedForegroundArea++;

      /** Update the moving foreground value. */
      if ( movingValue == this->m_ForegroundValue ) movingForegroundArea++;

      /** Update the intersection. */
      if ( movingValue == this->m_ForegroundValue && fixedValue == this->m_ForegroundValue )
      {        
        intersection++;
      }

      /** Update the NumberOfPixelsCounted. */
      this->m_NumberOfPixelsCounted++;

    } // end if IsInsideBuffer()

  } // end for loop over the image sample container

  /** Throw exceptions if necessary. */
	if ( this->m_NumberOfPixelsCounted == 0 )
	{
		itkExceptionMacro( << "All the points mapped outside the moving image" );
	}

  /** Compute the final metric value. */
  measure = 2.0 * intersection / ( fixedForegroundArea + movingForegroundArea );
  if ( this->m_Complement )
  { 
    measure = 1.0 - measure;
  }

  /** Return the kappa statistic value. */
  return measure;

} // end GetValue()


/**
 * ******************* GetDerivative *******************
 */

template < class TFixedImage, class TMovingImage> 
void
KappaStatisticImageToImageMetric2<TFixedImage,TMovingImage>
::GetDerivative( const TransformParametersType & parameters,
  DerivativeType & derivative  ) const
{
  /** When the derivative is calculated, all information for calculating
	 * the metric value is available. It does not cost anything to calculate
   * the metric value now. Therefore, we have chosen to only implement the
   * GetValueAndDerivative(), supplying it with a dummy value variable. */
  MeasureType dummyvalue = NumericTraits< MeasureType >::Zero;
  this->GetValueAndDerivative( parameters, dummyvalue, derivative );

} // end GetDerivative()


/**
 * ******************* GetValueAndDerivative *******************
 */

template < class TFixedImage, class TMovingImage> 
void
KappaStatisticImageToImageMetric2<TFixedImage,TMovingImage>
::GetValueAndDerivative( const TransformParametersType & parameters,
  MeasureType & value, DerivativeType & derivative  ) const
{
  itkDebugMacro( "GetValueAndDerivative( " << parameters << " ) ");

  /** Some sanity checks. */
  if ( !this->GetGradientImage() )
  {
    itkExceptionMacro( << "The gradient image is null, maybe you forgot to call Initialize()" );
  }

  /** Make sure the transform parameters are up to date. */
  this->SetTransformParameters( parameters );
  const unsigned int ParametersDimension = this->GetNumberOfParameters();

  this->m_NumberOfPixelsCounted = 0;

  /** Update the imageSampler and get a handle to the sample container. */
  this->GetImageSampler()->Update();
  ImageSampleContainerPointer sampleContainer = this->GetImageSampler()->GetOutput();

  /** Create iterator over the sample container. */
  typename ImageSampleContainerType::ConstIterator iter;
  typename ImageSampleContainerType::ConstIterator begin = sampleContainer->Begin();
  typename ImageSampleContainerType::ConstIterator end = sampleContainer->End();

  /** Some typedefs. */
  typedef typename OutputPointType::CoordRepType	CoordRepType;
  typedef ContinuousIndex<CoordRepType,
    MovingImageType::ImageDimension>							MovingImageContinuousIndexType;

  /** Create variables to store intermediate results.
   * 'measure' is the value of the metric.
   * 'fixedForegroundArea' is the total area of the foreground region
   *    in the fixed image.
   * 'movingForegroundArea' is the foreground area in the moving image
   *    in the area of overlap under the current transformation.
   * 'intersection' is the area of foreground intersection between the
   *    fixed and moving image.
   */
  MeasureType measure              = NumericTraits< MeasureType >::Zero;
  MeasureType fixedForegroundArea  = NumericTraits< MeasureType >::Zero;
  MeasureType movingForegroundArea = NumericTraits< MeasureType >::Zero;
  MeasureType intersection         = NumericTraits< MeasureType >::Zero;

  //typename FixedImageType::IndexType index;
  typedef Array<double> ArrayType;
  ArrayType sum1 = ArrayType( ParametersDimension );
  sum1.Fill(  NumericTraits<ITK_TYPENAME ArrayType::ValueType>::Zero );
  ArrayType sum2 = ArrayType( ParametersDimension );
  sum2.Fill(  NumericTraits<ITK_TYPENAME ArrayType::ValueType>::Zero );

  InputPointType	                    inputPoint;
  OutputPointType                     transformedPoint;
  MovingImageContinuousIndexType			tempIndex;
  typename MovingImageType::IndexType	mappedIndex;

  derivative = DerivativeType( ParametersDimension );
  derivative.Fill( NumericTraits<ITK_TYPENAME DerivativeType::ValueType>::Zero );

  /** Loop over the fixed image to calculate the mean squares. */
  for ( iter = begin; iter != end; ++iter )
  {
    /** Get the current inputpoint. */
    inputPoint = (*iter).Value().m_ImageCoordinates;

    /** Transform the inputpoint to get the transformed point. */
    transformedPoint = this->m_Transform->TransformPoint( inputPoint );

    /** Inside the moving image mask? */
    if ( this->m_MovingImageMask && !this->m_MovingImageMask->IsInside( transformedPoint ) )
    {
      continue;
    }

    /** In this if-statement the actual calculation of the kappa statistic is done. */
    if ( this->m_Interpolator->IsInsideBuffer( transformedPoint ) )
    {
      /** Get the fixedValue = f(x) and the movingValue = m(x+u(x)). */
      const RealType movingValue  = this->m_Interpolator->Evaluate( transformedPoint );
      const RealType & fixedValue = (*iter).Value().m_ImageValue;

      /** Update the fixed foreground value. */
      if ( fixedValue == this->m_ForegroundValue ) fixedForegroundArea++;

      /** Update the moving foreground value. */
      if ( movingValue == this->m_ForegroundValue ) movingForegroundArea++;

      /** Update the intersection. */
      if ( fixedValue == this->m_ForegroundValue
        && movingValue == this->m_ForegroundValue )
      {        
        intersection++;
      }

      /** Get the Jacobian. */
      const TransformJacobianType & jacobian =
        this->m_Transform->GetJacobian( inputPoint ); 

      /** Get the gradient by NearestNeighboorInterpolation:
       * which is equivalent to round up the point components. */
      this->m_MovingImage->TransformPhysicalPointToContinuousIndex( transformedPoint, tempIndex );
      for ( unsigned int j = 0; j < MovingImageDimension; j++ )
      {
        mappedIndex[ j ] = static_cast<long>( vnl_math_rnd( tempIndex[ j ] ) );
      }
      const GradientPixelType gradient = this->GetGradientImage()->GetPixel( mappedIndex );

      /** Calculate the contributions to all parameters. */
      for ( unsigned int par = 0; par < ParametersDimension; par++ )
      {
        for ( unsigned int dim = 0; dim < FixedImageDimension; dim++ )
        {
          sum2[ par ] += jacobian( dim, par ) * gradient[ dim ];
          if ( fixedValue == this->m_ForegroundValue )
          {
            sum1[ par ] += 2.0 * jacobian( dim, par ) * gradient[ dim ];
          }            
        }
      }

      /** Update the NumberOfPixelsCounted. */
      this->m_NumberOfPixelsCounted++;

    } // end if IsInsideBuffer()

  } // end for loop over the image sample container

  /** Throw exceptions if necessary. */
  if ( this->m_NumberOfPixelsCounted == 0 )
  {
    itkExceptionMacro( << "All the points mapped outside the moving image" );
  }

  /** Compute the final metric value. */
  MeasureType areaSum = fixedForegroundArea + movingForegroundArea;
  measure = 2.0 * intersection / areaSum;
  if ( this->m_Complement )
  { 
    measure = 1.0 - measure;
  }
	value = measure;

  /** Calculate the derivative. */
  if ( !this->m_Complement )
  {
    for ( unsigned int par = 0; par < ParametersDimension; par++ )
    {
      derivative[ par ] = -( areaSum * sum1[ par ] - 2.0 * intersection * sum2[ par ] )
        / ( areaSum * areaSum );
    }
  }
  else
  {
    for ( unsigned int par = 0; par < ParametersDimension; par++ )
    {
      derivative[ par ] = ( areaSum * sum1[ par ] - 2.0 * intersection * sum2[ par ] )
        / ( areaSum * areaSum );
    }
  }

} // end GetValueAndDerivative()


/*
 * Compute the image gradient and assign to m_GradientImage.
 * Overrides superclass implementation
 *
template <class TFixedImage, class TMovingImage> 
void
KappaStatisticImageToImageMetric2<TFixedImage,TMovingImage>
::ComputeGradient()
{
  const unsigned int dim = MovingImageType::ImageDimension;

  typedef itk::Image< GradientPixelType, dim > GradientImageType;
  typename GradientImageType::Pointer tempGradientImage = GradientImageType::New();
    tempGradientImage->SetRegions( this->m_MovingImage->GetBufferedRegion().GetSize() );
    tempGradientImage->Allocate();
    tempGradientImage->Update();

  typedef  itk::ImageRegionIteratorWithIndex< GradientImageType > GradientIteratorType;
  typedef  itk::ImageRegionConstIteratorWithIndex< MovingImageType > MovingIteratorType; 

  GradientIteratorType git( tempGradientImage, tempGradientImage->GetBufferedRegion() );
  MovingIteratorType mit( this->m_MovingImage, this->m_MovingImage->GetBufferedRegion() );

  git.GoToBegin();
  mit.GoToBegin();

  typename MovingImageType::IndexType minusIndex;
  typename MovingImageType::IndexType plusIndex;
  typename MovingImageType::IndexType currIndex;
  typename GradientImageType::PixelType tempGradPixel;
  typename MovingImageType::SizeType movingSize = this->m_MovingImage->GetBufferedRegion().GetSize();
  while(!mit.IsAtEnd())
    {
    currIndex = mit.GetIndex();
    minusIndex = mit.GetIndex();
    plusIndex = mit.GetIndex();
    for ( unsigned int i=0; i<dim; i++ )
      {
      if ((currIndex[i] == 0)||(currIndex[i]==(movingSize[i]-1)))
        {
        tempGradPixel[i] = 0;
        }
      else
        {
        minusIndex[i] = currIndex[i]-1;
        plusIndex[i] = currIndex[i]+1;
        double minusVal = double(this->m_MovingImage->GetPixel(minusIndex));
        double val      = double(this->m_MovingImage->GetPixel(currIndex));
        double plusVal  = double(this->m_MovingImage->GetPixel(plusIndex));
        if ((minusVal != m_ForegroundValue)&&(plusVal == m_ForegroundValue))
          {
          tempGradPixel[i] = 1;
          }
        else if ((minusVal == m_ForegroundValue)&&(plusVal != m_ForegroundValue))
          {
          tempGradPixel[i] = -1;
          }
        else
          {
          tempGradPixel[i] = 0;
          }
        }
      minusIndex = currIndex;
      plusIndex  = currIndex;
      }
    git.Set( tempGradPixel );
    ++git;
    ++mit;
    }

  this->m_GradientImage = tempGradientImage;
} // end ComputeGradient()*/


} // end namespace itk


#endif // end #ifndef _itkKappaStatisticImageToImageMetric2_txx
