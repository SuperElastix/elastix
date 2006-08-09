#ifndef _itkAdvancedMeanSquaresImageToImageMetric_txx
#define _itkAdvancedMeanSquaresImageToImageMetric_txx

#include "itkAdvancedMeanSquaresImageToImageMetric.h"
#include "itkBoxSpatialObject.h"


namespace itk
{

	/**
	* ******************* Constructor *******************
	*/

	template <class TFixedImage, class TMovingImage> 
		AdvancedMeanSquaresImageToImageMetric<TFixedImage,TMovingImage>
		::AdvancedMeanSquaresImageToImageMetric()
	{
    //this->ComputeGradientOff();

    this->m_InternalMovingImageMask = 0;
    
	} // end constructor


  /**
	 * ********************* Initialize ****************************
	 */

  template <class TFixedImage, class TMovingImage>
    void
    AdvancedMeanSquaresImageToImageMetric<TFixedImage,TMovingImage>
    ::Initialize(void) throw ( ExceptionObject )
  {
    /** Initialize transform, interpolator, etc. */
    Superclass::Initialize();

    /** Initialize the derivative operators */
    this->InitializeDerivativeOperators();

    /** Initialize the internal moving image mask */
    this->InitializeInternalMasks();
    
    /** Prepare some stuff for computing derivatives on the internal masks */
    this->InitializeNeighborhoodOffsets();
    
  } // end Initialize


  /**
	 * ********************* InitializeDerivativeOperators *****************
	 */

  template <class TFixedImage, class TMovingImage>
    void
    AdvancedMeanSquaresImageToImageMetric<TFixedImage,TMovingImage>
    ::InitializeDerivativeOperators(void)
  {
    MovingImageSpacingType  spacing = this->GetMovingImage()->GetSpacing();

    DefaultMovingMaskDerivativeOperatorType defaultDerivativeOperator;
    defaultDerivativeOperator.SetOrder(1);
    for (unsigned int i = 0; i < MovingImageDimension; ++i)
    {
      defaultDerivativeOperator.SetDirection(i);
      defaultDerivativeOperator.CreateDirectional();
      defaultDerivativeOperator.FlipAxes();
      defaultDerivativeOperator.ScaleCoefficients( 1.0 / spacing[i] );
      this->SetMovingMaskDerivativeOperator(i, defaultDerivativeOperator );
    }
  } // end InitializeDerivativeOperator


  /**
	 * ********************* InitializeInternalMasks *********************
	 */

  template <class TFixedImage, class TMovingImage>
    void
    AdvancedMeanSquaresImageToImageMetric<TFixedImage,TMovingImage>
    ::InitializeInternalMasks(void)
  {
    /** Initialize the internal moving image mask */

    typedef itk::BoxSpatialObject<MovingImageDimension>          BoxType;
    typedef typename BoxType::SizeType                           BoxSizeType;
    typedef typename BoxType::TransformType                      SpatialObjectTransformType;
    typedef typename SpatialObjectTransformType::OffsetType      SpatialObjectOffsetType;
    typedef typename MovingImageType::PointType                  MovingImageOriginType;
    typedef typename MovingMaskDerivativeOperatorType::SizeType  SizeType;
    
    this->m_InternalMovingImageMask = 
      const_cast<MovingImageMaskType *>( this->GetMovingImageMask() );    

    if ( this->m_InternalMovingImageMask.IsNull() )
    {
      BoxType::Pointer box = BoxType::New();

      /** Determine max derivative operator radius for each dimension */
      SizeType derivopMaxRadius;
      derivopMaxRadius.Fill(0);
      /** Loop over all derivative operators */
      for ( unsigned int d1 = 0; d1 < MovingImageDimension; ++d1)
      {
        /** Loop over the dimensions of this derivative operator */
        for ( unsigned int d2 = 0; d2 < MovingImageDimension; ++d2)
        {
          const unsigned int derivopRadius_d2 = static_cast<unsigned int>(
            this->GetMovingMaskDerivativeOperator(d1).GetRadius(d2) );
          derivopMaxRadius[d2] = vnl_math_max( 
            derivopRadius_d2, static_cast<unsigned int>(derivopMaxRadius[d2]) );
        }
      }
    
      /** define the box its size and position */
      SizeType movingImageSize =
        this->GetMovingImage()->GetLargestPossibleRegion().GetSize();
      BoxSizeType boxSize;
      SpatialObjectOffsetType offset; 
      MovingImageOriginType   origin = this->GetMovingImage()->GetOrigin(); 
      MovingImageSpacingType  spacing = this->GetMovingImage()->GetSpacing(); 
      for( unsigned int d=0; d < MovingImageDimension; d++) 
      { 
        /** Size is a continuous offset; points on the edge are considered Inside */
        boxSize[d] = movingImageSize[d] - 2 * derivopMaxRadius[d] - 1;
        offset[d] = origin[d] + derivopMaxRadius[d] * spacing[d]; 
      } 
      box->SetSize( boxSize );
      box->SetSpacing( spacing.GetDataPointer() );
      box->GetIndexToObjectTransform()->SetOffset( offset );
      box->ComputeObjectToWorldTransform(); 

      /** Is this necessary? I copied it from the SetImage implementation of ImageSpatialObject */
      box->ComputeBoundingBox(); 

      /** Assign the box to the m_InternalMovingImageMask */
      this->m_InternalMovingImageMask = box.GetPointer();

      //test: 
      OutputPointType midden;
      OutputPointType neterin;
      OutputPointType neteruit;
      OutputPointType neternietuit;
      midden.Fill(128.0);
      neterin.Fill(1.0);
      neteruit.Fill(255.0);
      neternietuit.Fill(254.0);

    } // end if no moving mask

    

  } // end InitializeInternalMasks


  /**
	 * ********************* InitializeNeighborhoodOffsets *********************
   * Prepare stuff for computing derivatives on the moving mask
	 */

  template <class TFixedImage, class TMovingImage>
    void
    AdvancedMeanSquaresImageToImageMetric<TFixedImage,TMovingImage>
    ::InitializeNeighborhoodOffsets(void)
  {

    typedef typename MovingMaskDerivativeOperatorType::OffsetType 
      MovingDiscreteOffsetType;
    MovingRealOffsetType spacing = this->GetMovingImage()->GetSpacing();

    for (unsigned int d = 0; d < MovingImageDimension; ++d)
    {
      /** For readability: some aliases */
      MovingMaskNeighborhoodOffsetsType & realoffsets =
        this->m_MovingMaskNeighborhoodOffsetsArray[d];
      const MovingMaskDerivativeOperatorType & derivop =
        this->GetMovingMaskDerivativeOperator(d);
              
      /** Set the size of the realoffsets neighborhood */
      realoffsets.SetRadius( derivop.GetRadius() );
      
      /** populate the realoffsets neighborhood with physical offsets 
       * to the center element. */
      for (unsigned int i = 0; i < realoffsets.Size(); ++i)
      {
        /** The discrete offset of this neighborhood element */
        const MovingDiscreteOffsetType & discreteOffset = 
          derivop.GetOffset(i);
        /** Convert it to an offset in physical spacing */
        for (unsigned j = 0; j < MovingImageDimension; ++j)
        {
          realoffsets[i][j] = discreteOffset[j] * spacing[j];
        } // end for j
      } // end for i

    } // end for d

  } // end InitializeNeighborhoodOffsets


	/**
	 * **************** EvaluateMovingMaskValueAndDerivative *******************
   * Estimate value and spatial derivative of internal moving mask 
   */

	template < class TFixedImage, class TMovingImage> 
		void
		AdvancedMeanSquaresImageToImageMetric<TFixedImage,TMovingImage>
    ::EvaluateMovingMaskValueAndDerivative(
      const OutputPointType & point,
      double & value,
      MovingMaskDerivativeType & derivative) const
  {
    /** get the internal mask */
    typename MovingImageMaskType::ConstPointer movingMask =
      this->GetInternalMovingImageMask();

    /** Compute the value of the mask */
    movingMask->ValueAt(point, value);

    /** compute the spatial derivative in each dimension */
    derivative.Fill(0.0);
    for ( unsigned int d = 0; d < MovingImageDimension; ++d)
    {
      /** For readability: some aliases */
      const MovingMaskNeighborhoodOffsetsType & realoffsets =
        this->m_MovingMaskNeighborhoodOffsetsArray[d];
      const MovingMaskDerivativeOperatorType & derivop =
        this->GetMovingMaskDerivativeOperator(d);
      
      /** Calculate inner product of mask neighbourhood with derivative operator */
      double derivative_d = 0.0;
      for (unsigned int i = 0; i < realoffsets.Size(); ++i)
      {
         OutputPointType currentPoint = point + realoffsets[i];
         double currentValue;
         movingMask->ValueAt(currentPoint, currentValue);
         derivative_d += currentValue * derivop[i];
      }  
      derivative[d] = derivative_d;      
    }
    
  } // end EvaluateMovingMaskValueAndDerivative


	/**
	 * ******************* PrintSelf *******************
	 */

	template < class TFixedImage, class TMovingImage> 
		void
		AdvancedMeanSquaresImageToImageMetric<TFixedImage,TMovingImage>
		::PrintSelf(std::ostream& os, Indent indent) const
	{
		Superclass::PrintSelf( os, indent );

	} // end PrintSelf


	/**
	 * ******************* GetValue *******************
	 */

	template <class TFixedImage, class TMovingImage> 
		typename AdvancedMeanSquaresImageToImageMetric<TFixedImage,TMovingImage>::MeasureType
		AdvancedMeanSquaresImageToImageMetric<TFixedImage,TMovingImage>
		::GetValue( const TransformParametersType & parameters ) const
	{
		itkDebugMacro( "GetValue( " << parameters << " ) " );

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

		/** Create variables to store intermediate results. */
		InputPointType  inputPoint;
		OutputPointType transformedPoint;

		MeasureType measure = NumericTraits< MeasureType >::Zero;

		/** Loop over the fixed image samples to calculate the mean squares. */
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

			/** In this if-statement the actual calculation of mean squares is done. */
			if ( this->m_Interpolator->IsInsideBuffer( transformedPoint ) )
			{
				/** Get the fixedValue = f(x) and the movingValue = m(x+u(x)). */
				const RealType movingValue  = this->m_Interpolator->Evaluate( transformedPoint );
        const RealType & fixedValue = (*iter).Value().m_ImageValue;

				/** The difference squared. */
				const RealType diff = movingValue - fixedValue; 
				measure += diff * diff;

				/** Update the NumberOfPixelsCounted. */
				this->m_NumberOfPixelsCounted++;

			} // end if IsInsideBuffer()

		} // end for loop over the image sample container

		/** Calculate the measure value. */
		if ( this->m_NumberOfPixelsCounted > 0 )
		{
			measure /= this->m_NumberOfPixelsCounted;
		}
		else
		{
			measure = NumericTraits< MeasureType >::Zero;
		}

		/** Throw exceptions if necessary. */
		if ( this->m_NumberOfPixelsCounted == 0 )
		{
			itkExceptionMacro( << "All the points mapped outside the moving image" );
		}

		/** Return the mean squares measure value. */
		return measure;

	} // end GetValue
	

	/**
	 * ******************* GetDerivative *******************
	 */

	template < class TFixedImage, class TMovingImage> 
		void
		AdvancedMeanSquaresImageToImageMetric<TFixedImage,TMovingImage>
		::GetDerivative( const TransformParametersType & parameters,
		DerivativeType & derivative ) const
	{
		/** When the derivative is calculated, all information for calculating
		 * the metric value is available. It does not cost anything to calculate
		 * the metric value now. Therefore, we have chosen to only implement the
		 * GetValueAndDerivative(), supplying it with a dummy value variable. */
		MeasureType dummyvalue = NumericTraits< MeasureType >::Zero;
		this->GetValueAndDerivative( parameters, dummyvalue, derivative );

	} // end GetDerivative


	/**
	 * ******************* GetValueAndDerivative *******************
	 */

	template <class TFixedImage, class TMovingImage>
		void
		AdvancedMeanSquaresImageToImageMetric<TFixedImage,TMovingImage>
		::GetValueAndDerivative( const TransformParametersType & parameters, 
		MeasureType & value, DerivativeType & derivative ) const
	{
		itkDebugMacro("GetValueAndDerivative( " << parameters << " ) ");

    typedef typename DerivativeType::ValueType        DerivativeValueType;
    typedef typename TransformJacobianType::ValueType TransformJacobianValueType;
 
		/** Some sanity checks. */
		if ( !this->GetGradientImage() )
		{
			itkExceptionMacro( << "The gradient image is null, maybe you forgot to call Initialize()" );
		}

		/** Make sure the transform parameters are up to date. */
		this->SetTransformParameters( parameters );
		const unsigned int ParametersDimension = this->GetNumberOfParameters();
		
		this->m_NumberOfPixelsCounted = 0;
    double normalizationFactor = 0.0;

    /** Update the imageSampler and get a handle to the sample container. */
    this->GetImageSampler()->Update();
    ImageSampleContainerPointer sampleContainer = this->GetImageSampler()->GetOutput();

    /** Create iterator over the sample container. */
    typename ImageSampleContainerType::ConstIterator iter;
    typename ImageSampleContainerType::ConstIterator begin = sampleContainer->Begin();
    typename ImageSampleContainerType::ConstIterator end = sampleContainer->End();

		/** Some typedefs. */
		typedef typename OutputPointType::CoordRepType	CoordRepType;
		
    /** Create variable to store intermediate results. */
		typename MovingImageType::IndexType	mappedIndex;

		MeasureType measure = NumericTraits< MeasureType >::Zero;
		
		derivative = DerivativeType( ParametersDimension );
		derivative.Fill( NumericTraits< DerivativeValueType >::Zero );

    DerivativeType ddendmu( ParametersDimension );
    DerivativeType dnumdmu( ParametersDimension );
    ddendmu.Fill( NumericTraits< DerivativeValueType >::Zero );
    dnumdmu.Fill( NumericTraits< DerivativeValueType >::Zero );

		/** Loop over the fixed image to calculate the mean squares. */
		for ( iter = begin; iter != end; ++iter )
		{
			/** Get the current inputpoint. */
      const InputPointType & inputPoint = (*iter).Value().m_ImageCoordinates;

			/** Transform the inputpoint to get the transformed point. */
			const OutputPointType transformedPoint = this->m_Transform->TransformPoint( inputPoint );

      double movingMaskValue = 0.0;
      MovingMaskDerivativeType movingMaskDerivative; 
      this->EvaluateMovingMaskValueAndDerivative(
        transformedPoint, movingMaskValue, movingMaskDerivative);
      const double movingMaskDerivativeMagnitude = movingMaskDerivative.GetNorm();
      const double smallNumber1 = 1e-10;

			/** Inside the moving image mask? */
			if ( movingMaskValue < smallNumber1 && movingMaskDerivativeMagnitude < smallNumber1)
			{
				continue;
			}

			/** In this if-statement the actual calculation of mean squares is done. */
			if ( this->m_Interpolator->IsInsideBuffer( transformedPoint ) )
			{
				/** Get the fixedValue = f(x) and the movingValue = m(x+u(x)). */
				const RealType movingValue  = this->m_Interpolator->Evaluate( transformedPoint );
        const RealType & fixedValue = (*iter).Value().m_ImageValue;

				/** Get the Jacobian. */
				const TransformJacobianType & jacobian =
					this->m_Transform->GetJacobian( inputPoint ); 

				/** The difference squared. */
				const RealType diff = movingValue - fixedValue; 
        const RealType diffdiff = diff * diff;
				measure += movingMaskValue * diffdiff;
        normalizationFactor += movingMaskValue;

				/** Get the gradient by NearestNeighboorInterpolation:
				 * which is equivalent to round up the point components.
         * The itk::ImageFunction provides a function to do this.
         */
				this->m_Interpolator->ConvertPointToNearestIndex( transformedPoint, mappedIndex );
				const GradientPixelType gradient = this->GetGradientImage()->GetPixel( mappedIndex );
        
				/** Calculate the contributions to the derivatives with respect to each parameter. */
        const RealType movmask_diff_2 = movingMaskValue * diff * 2.0;
				for ( unsigned int par = 0; par < ParametersDimension; par++ )
				{
          /** compute inproduct of image gradient and transform jacobian */
					RealType grad_jac = NumericTraits< RealType >::Zero;
          RealType maskderiv_jac = NumericTraits< RealType >::Zero;
          for( unsigned int dim = 0; dim < MovingImageDimension; dim++ )
					{
            const TransformJacobianValueType & jacdimpar = jacobian( dim, par );
						grad_jac += static_cast<RealType>(
              gradient[ dim ] * jacdimpar );
            maskderiv_jac += static_cast<RealType> (
              movingMaskDerivative[ dim ] * jacdimpar );
					}
          dnumdmu[ par ] += movmask_diff_2 * grad_jac + diffdiff * maskderiv_jac;
          ddendmu[ par ] += maskderiv_jac;
				}
        
				/** Update the NumberOfPixelsCounted. */
				this->m_NumberOfPixelsCounted++;

			} // end if IsInsideBuffer()

		} // end for loop over the image sample container

		/** Calculate the value and the derivative. */
    const double smallNumber2 = 1e-10;
		if ( this->m_NumberOfPixelsCounted > 0 &&  normalizationFactor > smallNumber2 )
		{
			measure /= normalizationFactor;
      MeasureType measure_N = measure / normalizationFactor;
      for( unsigned int i = 0; i < ParametersDimension; i++ )
			{
				derivative[ i ] = dnumdmu[i] / normalizationFactor - ddendmu[i] * measure_N ;
			}
		}
		else
		{
			measure = NumericTraits< MeasureType >::Zero;
			derivative.Fill( NumericTraits<ITK_TYPENAME DerivativeType::ValueType>::Zero );
			itkExceptionMacro( << "All the points mapped outside the moving image" );
		}

		/** The return value. */
		value = measure;

	} // end GetValueAndDerivative


} // end namespace itk


#endif // end #ifndef _itkAdvancedMeanSquaresImageToImageMetric_txx

