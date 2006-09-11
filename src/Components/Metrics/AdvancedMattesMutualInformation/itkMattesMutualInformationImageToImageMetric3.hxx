#ifndef _itkMattesMutualInformationImageToImageMetric3_HXX__
#define _itkMattesMutualInformationImageToImageMetric3_HXX__

#include "itkMattesMutualInformationImageToImageMetric3.h"

#include "itkImageRegionConstIterator.h"
#include "itkImageRegionIterator.h"

#include "itkImageRegionExclusionIteratorWithIndex.h"
#include "itkImageRegionIteratorWithIndex.h"
#include "itkBinaryErodeImageFilter.h"
#include "itkBinaryBallStructuringElement.h"
#include "itkImageLinearIteratorWithIndex.h"
#include "itkImageSliceIteratorWithIndex.h"
#include "vnl/vnl_math.h"


namespace itk
{
	
	
	/**
	 * ********************* Constructor ****************************
	 */

	template < class TFixedImage, class TMovingImage >
		MattesMutualInformationImageToImageMetric3<TFixedImage,TMovingImage>
		::MattesMutualInformationImageToImageMetric3()
	{
		this->SetComputeGradient(false); // don't use the default gradient

    this->m_NumberOfFixedHistogramBins = 50;
    this->m_NumberOfMovingHistogramBins = 50;
		this->m_JointPDF = 0;
		this->m_JointPDFDerivatives = 0;
    this->m_FixedImageNormalizedMin = 0.0;
		this->m_MovingImageNormalizedMin = 0.0;
		this->m_FixedImageTrueMin = 0.0;
		this->m_FixedImageTrueMax = 0.0;
		this->m_MovingImageTrueMin = 0.0;
		this->m_MovingImageTrueMax = 0.0;
		this->m_FixedImageBinSize = 0.0;
		this->m_MovingImageBinSize = 0.0;

    this->m_FixedKernel = 0;
    this->m_MovingKernel = 0;
		this->m_DerivativeMovingKernel = 0;
    this->m_FixedKernelBSplineOrder = 0;
    this->m_MovingKernelBSplineOrder = 3;
   
    this->m_CheckNumberOfSamples = true;

		this->m_BSplineInterpolator = 0;
		this->m_DerivativeCalculator = 0;
    this->m_InterpolatorIsBSpline = false;
		
		this->m_BSplineTransform = 0;
		this->m_BSplineCombinationTransform = 0;
    this->m_NumParametersPerDim = 0;
		this->m_NumBSplineWeights = 0;
		this->m_NumberOfParameters = 0;
    this->m_TransformIsBSpline = false;
		this->m_TransformIsBSplineCombination = false;
 
    const unsigned int defaultMaskInterpolationOrder = 2;
    this->m_InternalMovingImageMask = 0;
    this->m_MovingImageMaskInterpolator = 
      MovingImageMaskInterpolatorType::New();
    this->m_MovingImageMaskInterpolator->SetSplineOrder( defaultMaskInterpolationOrder );
    this->m_UseDifferentiableOverlap = true;

    this->m_HardLimitMovingGrayValues = false;
    this->m_SoftLimitMovingGrayValues = true;
    this->m_FixedLimitRangeRatio = 0.01;
    this->m_MovingLimitRangeRatio = 0.01;
    this->m_SoftMaxLimit_a = 0.0;
    this->m_SoftMaxLimit_A = 0.0;
    this->m_SoftMinLimit_a = 0.0;
    this->m_SoftMinLimit_A = 0.0;
    this->m_MovingImageMinLimit = 0.0;
    this->m_MovingImageMaxLimit = 1.0;
    this->m_FixedImageMinLimit = 0.0;
    this->m_MovingImageMaxLimit = 1.0;

    this->m_FixedParzenTermToIndexOffset = 0.5;
    this->m_MovingParzenTermToIndexOffset = -1.0;
   		
	} // end Constructor
	
	

	/**
	 * ********************* PrintSelf ******************************
	 *
	 * Print out internal information about this class.
	 */

	template < class TFixedImage, class TMovingImage  >
		void
		MattesMutualInformationImageToImageMetric3<TFixedImage,TMovingImage>
		::PrintSelf( std::ostream& os, Indent indent ) const
	{
		/** Call the superclass' PrintSelf. */
		Superclass::PrintSelf( os, indent );
		
		/** Add debugging information. */
		os << indent << "NumberOfHistogramBins: ";
		os << this->m_NumberOfFixedHistogramBins << std::endl;
    os << this->m_NumberOfMovingHistogramBins << std::endl;

    /** This function is not complete, but we don't use it anyway. */
		
	} // end PrintSelf
	
	
	/**
	 * ********************* Initialize *****************************
	 */

	template <class TFixedImage, class TMovingImage> 
		void
		MattesMutualInformationImageToImageMetric3<TFixedImage,TMovingImage>
		::Initialize(void) throw ( ExceptionObject )
	{
		/** Call the superclass to check that standard components are available */
		this->Superclass::Initialize();
		
		/** Cache the number of transformation parameters. This line 
     * emphasises that a user has to call Initialize again if the number
     * of parameters is changed (since the histograms have to be resized) */
		this->m_NumberOfParameters = this->m_Transform->GetNumberOfParameters();
		
    /** Compute the max and min of the fixed and moving images */
    this->ComputeImageExtrema(
      this->m_FixedImageTrueMin,
      this->m_FixedImageTrueMax,
      this->m_MovingImageTrueMin,
      this->m_MovingImageTrueMax );

    /** Setup the parameters for the gray value limiter */
    this->InitializeLimiter();

    /** Set up the histograms */
    this->InitializeHistograms();

    /** Set up the Parzen windows */
    this->InitializeKernels();

    /** Check if the interpolator is a bspline interpolator */
    this->CheckForBSplineInterpolator();

    /** Check if the transform is a BSplineTransform or a BSplineCombinationTransform */
    this->CheckForBSplineTransform();

    /** Allocate memory for the alpha derivatives, and innerproducts */
    this->m_AlphaDerivatives.SetSize( this->m_NumberOfParameters );
    
    /** Initialize the internal moving image mask */
    this->InitializeInternalMasks();
	
	} // end Initialize


  /**
	 * ****************** ComputeImageExtrema ***************************
	 */

	template <class TFixedImage, class TMovingImage> 
		void
		MattesMutualInformationImageToImageMetric3<TFixedImage,TMovingImage>
		::ComputeImageExtrema(
      double & fixedImageMin, double & fixedImageMax,
      double & movingImageMin, double & movingImageMax) const
  {
 		/** Compute the minimum and maximum for the FixedImage over
		 * the FixedImageRegion.
		 * NB: We can't use StatisticsImageFilter to do this because
		 * the filter computes the min/max for the largest possible region. */
		double fixedImageMinTemp = NumericTraits<double>::max();
		double fixedImageMaxTemp = NumericTraits<double>::NonpositiveMin();

		typedef ImageRegionConstIterator<FixedImageType> FixedIteratorType;
		FixedIteratorType fixedImageIterator( 
			this->m_FixedImage, this->GetFixedImageRegion() );

		for ( fixedImageIterator.GoToBegin(); !fixedImageIterator.IsAtEnd();
      ++fixedImageIterator )
    {
			double sample = static_cast<double>( fixedImageIterator.Get() );
      fixedImageMinTemp = vnl_math_min( fixedImageMinTemp, sample );
      fixedImageMaxTemp = vnl_math_max( fixedImageMaxTemp, sample );
    }
		
		/** Compute the minimum and maximum for the entire moving image
		 * in the buffer. */
		double movingImageMinTemp = NumericTraits<double>::max();
		double movingImageMaxTemp = NumericTraits<double>::NonpositiveMin();
		
		typedef ImageRegionConstIterator<MovingImageType> MovingIteratorType;
		MovingIteratorType movingImageIterator(
			this->m_MovingImage, this->m_MovingImage->GetBufferedRegion() );
		
		for ( movingImageIterator.GoToBegin(); 
		!movingImageIterator.IsAtEnd(); ++movingImageIterator)
    {
			double sample = static_cast<double>( movingImageIterator.Get() );
      movingImageMinTemp = vnl_math_min( movingImageMinTemp, sample );
      movingImageMaxTemp = vnl_math_max( movingImageMaxTemp, sample );
		}

    fixedImageMin = fixedImageMinTemp;
    fixedImageMax = fixedImageMaxTemp;
    movingImageMin = movingImageMinTemp;
    movingImageMax = movingImageMaxTemp;
  } // end ComputeImageExtrema


  /**
   * ****************** InitializeLimiter *****************************
   * in: the image extrema
   * out: the parameters for the gray value limiter and the extremal
   * values of the limiter output ( m_{Fixed,Moving}Image{Max,Min}Limit );
   * These last variables are needed for the histogram size definition.
	 */

	template <class TFixedImage, class TMovingImage> 
		void
		MattesMutualInformationImageToImageMetric3<TFixedImage,TMovingImage>
		::InitializeLimiter(void)
  {
    /** We assume that the image does not contain values close to 
     * the max(double) or min(double). */
    this->m_MovingImageMaxLimit = 
      this->m_MovingImageTrueMax + this->m_MovingLimitRangeRatio *
      ( this->m_MovingImageTrueMax - this->m_MovingImageTrueMin );
    this->m_MovingImageMinLimit = 
      this->m_MovingImageTrueMin - this->m_MovingLimitRangeRatio *
      ( this->m_MovingImageTrueMax - this->m_MovingImageTrueMin );
    this->m_FixedImageMaxLimit = 
      this->m_FixedImageTrueMax + this->m_FixedLimitRangeRatio *
      ( this->m_FixedImageTrueMax - this->m_FixedImageTrueMin );
    this->m_FixedImageMinLimit = 
      this->m_FixedImageTrueMin - this->m_FixedLimitRangeRatio *
      ( this->m_FixedImageTrueMax - this->m_FixedImageTrueMin );
		
    /** Compute settings for the soft limiter */
    if ( (this->m_MovingImageTrueMax - this->m_MovingImageMaxLimit) < -1e-10 )
    {
      this->m_SoftMaxLimit_A = 
        this->m_MovingImageTrueMax - this->m_MovingImageMaxLimit;
      this->m_SoftMaxLimit_a = 1.0 / this->m_SoftMaxLimit_A;
    }
    else
    {
      /** The result is a hard limiter */
      this->m_SoftMaxLimit_a = 0.0;
      this->m_SoftMaxLimit_A = 0.0;
    }
    if ( (this->m_MovingImageTrueMin - this->m_MovingImageMinLimit) > 1e-10 )
    {
      this->m_SoftMinLimit_A = 
        this->m_MovingImageTrueMin - this->m_MovingImageMinLimit;
      this->m_SoftMinLimit_a = 1.0 / this->m_SoftMinLimit_A;
    }
    else
    {
      /** The result is a hard limiter */
      this->m_SoftMinLimit_a = 0.0;
      this->m_SoftMinLimit_A = 0.0;
    }
	
  } // end InitializeLimiter


  /**
   * ****************** InitializeHistograms *****************************
   */

	template <class TFixedImage, class TMovingImage> 
		void
		MattesMutualInformationImageToImageMetric3<TFixedImage,TMovingImage>
		::InitializeHistograms(void)
  {
  	/* Compute binsize for the histogram
     * 
		 * The binsize for the image intensities needs to be adjusted so that 
		 * we can avoid dealing with boundary conditions using the cubic 
		 * spline as the Parzen window.  We do this by increasing the size
		 * of the bins so that the joint histogram becomes "padded" at the 
		 * borders. Because we are changing the binsize, 
		 * we also need to shift the minimum by the padded amount in order to 
		 * avoid minimum values filling in our padded region.
		 *
		 * Note that there can still be non-zero bin values in the padded region,
		 * it's just that these bins will never be a central bin for the Parzen
		 * window.
     * \todo the padding should depend on the parzen window bspline order 
     * for comparison it is maybe more nice like this though.
     *
		 */
		//int fixedPadding = 2;  // this will pad by 2 bins
    //int movingPadding = 2;  // this will pad by 2 bins
    int fixedPadding = this->m_FixedKernelBSplineOrder / 2; // should be enough
    int movingPadding = this->m_MovingKernelBSplineOrder / 2;

    /** The ratio times the expected bin size will be added twice to the image range */
    const double smallNumberRatio = 0.001;
    const double smallNumberFixed = smallNumberRatio *
      ( this->m_FixedImageMaxLimit - this->m_FixedImageMinLimit ) /
			static_cast<double>( this->m_NumberOfFixedHistogramBins - 2 * fixedPadding - 1 );
    const double smallNumberMoving = smallNumberRatio *
      ( this->m_MovingImageMaxLimit - this->m_MovingImageMinLimit ) /
			static_cast<double>( this->m_NumberOfFixedHistogramBins - 2 * movingPadding - 1 );
    
    /** Compute binsizes */   		
		this->m_FixedImageBinSize = 
      ( this->m_FixedImageMaxLimit - this->m_FixedImageMinLimit + 2.0 * smallNumberFixed ) /
			static_cast<double>( this->m_NumberOfFixedHistogramBins - 2 * fixedPadding - 1 );
		this->m_FixedImageNormalizedMin = 
      (this->m_FixedImageMinLimit - smallNumberFixed ) / this->m_FixedImageBinSize
      - static_cast<double>( fixedPadding );
		
		this->m_MovingImageBinSize = 
      ( this->m_MovingImageMaxLimit - this->m_MovingImageMinLimit + 2.0 * smallNumberMoving ) /
			static_cast<double>( this->m_NumberOfMovingHistogramBins - 2 * movingPadding -1 );
		this->m_MovingImageNormalizedMin = 
      ( this->m_MovingImageMinLimit - smallNumberMoving ) / this->m_MovingImageBinSize
      - static_cast<double>( movingPadding );
				
		/** Allocate memory for the marginal PDF.	*/
		this->m_FixedImageMarginalPDF.SetSize( this->m_NumberOfFixedHistogramBins );
		this->m_MovingImageMarginalPDF.SetSize( this->m_NumberOfMovingHistogramBins );
    
		/** Allocate memory for the joint PDF and joint PDF derivatives. */
		this->m_JointPDF = JointPDFType::New();
		this->m_JointPDFDerivatives = JointPDFDerivativesType::New();
		JointPDFRegionType            jointPDFRegion;
		JointPDFIndexType             jointPDFIndex;
		JointPDFSizeType              jointPDFSize;
		JointPDFDerivativesRegionType jointPDFDerivativesRegion;
		JointPDFDerivativesIndexType  jointPDFDerivativesIndex;
		JointPDFDerivativesSizeType   jointPDFDerivativesSize;
		
		/** For the joint PDF define a region starting from {0,0} 
		 * with size {this->m_NumberOfMovingHistogramBins, this->m_NumberOfFixedHistogramBins}
		 * The dimension represents moving image parzen window index
	   * and fixed image parzen window index, respectively.
     * The moving parzen index is chosen as the first dimension,
     * because probably the moving bspline kernel order will be larger
     * than the fixed bspline kernel order and it is faster to iterate along
     * the first dimension   */
		jointPDFIndex.Fill( 0 ); 
		jointPDFSize[0] = this->m_NumberOfMovingHistogramBins; 
    jointPDFSize[1] = this->m_NumberOfFixedHistogramBins; 
		jointPDFRegion.SetIndex( jointPDFIndex );
		jointPDFRegion.SetSize( jointPDFSize );
		this->m_JointPDF->SetRegions( jointPDFRegion );
		this->m_JointPDF->Allocate();
		
		/** For the derivatives of the joint PDF define a region starting from {0,0,0} 
		 * with size {m_NumberOfParameters,m_NumberOfMovingHistogramBins, 
		 * m_NumberOfFixedHistogramBins}. The dimension represents transform parameters,
		 * moving image parzen window index and fixed image parzen window index,
		 * respectively. */
		jointPDFDerivativesIndex.Fill( 0 ); 
		jointPDFDerivativesSize[0] = this->m_NumberOfParameters;
		jointPDFDerivativesSize[1] = this->m_NumberOfMovingHistogramBins;
		jointPDFDerivativesSize[2] = this->m_NumberOfFixedHistogramBins;
		jointPDFDerivativesRegion.SetIndex( jointPDFDerivativesIndex );
		jointPDFDerivativesRegion.SetSize( jointPDFDerivativesSize );
		this->m_JointPDFDerivatives->SetRegions( jointPDFDerivativesRegion );
		this->m_JointPDFDerivatives->Allocate();

    
  } // end InitializeHistograms
				

  /**
   * ****************** InitializeKernels *****************************
   * Setup the kernels used for the Parzen windows.
	 */

	template <class TFixedImage, class TMovingImage> 
		void
		MattesMutualInformationImageToImageMetric3<TFixedImage,TMovingImage>
		::InitializeKernels(void)
  {
    switch ( this->m_FixedKernelBSplineOrder )
    {
      case 0:
        this->m_FixedKernel = BSplineKernelFunction<0>::New(); break;
      case 1:
        this->m_FixedKernel = BSplineKernelFunction<1>::New(); break;
      case 2:
        this->m_FixedKernel = BSplineKernelFunction<2>::New(); break;
      case 3:
        this->m_FixedKernel = BSplineKernelFunction<3>::New(); break;
      default:         
        itkExceptionMacro(<< "The following FixedKernelBSplineOrder is not implemented: "\
          << this->m_FixedKernelBSplineOrder );
    } // end switch FixedKernelBSplineOrder
    switch ( this->m_MovingKernelBSplineOrder )
    {
      case 0:
        this->m_MovingKernel = BSplineKernelFunction<0>::New();
        /** The derivative of a zero order bspline makes no sense. Using the
         * derivative of a first order gives a kind of finite difference idea
         * Anyway, if you plan to call GetValueAndDerivative you should use 
         * a higher bspline order. */
        this->m_DerivativeMovingKernel = BSplineDerivativeKernelFunction<1>::New();
        break;
      case 1:
        this->m_MovingKernel = BSplineKernelFunction<1>::New();
        this->m_DerivativeMovingKernel = BSplineDerivativeKernelFunction<1>::New();
        break;
      case 2:
        this->m_MovingKernel = BSplineKernelFunction<2>::New();
        this->m_DerivativeMovingKernel = BSplineDerivativeKernelFunction<2>::New();
        break;
      case 3:
        this->m_MovingKernel = BSplineKernelFunction<3>::New();
        this->m_DerivativeMovingKernel = BSplineDerivativeKernelFunction<3>::New();
        break;
      default:         
        itkExceptionMacro(<< "The following MovingKernelBSplineOrder is not implemented: "\
          << this->m_MovingKernelBSplineOrder );
    } // end switch MovingKernelBSplineOrder

    /** The region of support of the parzen window determines which bins
		* of the joint PDF are effected by the pair of image values.
		* For example, if we are using a cubic spline for the moving image parzen
		* window, four bins are affected. If the fixed image parzen window is
		* a zero-order spline (box car) only one bin is affected. */

    /** Set the size of the parzen window. */
    JointPDFSizeType parzenWindowSize;
    parzenWindowSize[0] = this->m_MovingKernelBSplineOrder + 1;
    parzenWindowSize[1] = this->m_FixedKernelBSplineOrder + 1;
    this->m_JointPDFWindow.SetSize( parzenWindowSize );
    this->m_JointPDFWindow.SetSize( parzenWindowSize );

    /** The ParzenIndex is the lowest bin number that is affected by a
     * pixel and computed as:
     * ParzenIndex = vcl_floor( ParzenTerm + ParzenTermToIndexOffset )
     * where ParzenTermToIndexOffset = 1/2, 0, -1/2, or -1  */
    this->m_FixedParzenTermToIndexOffset = 
      0.5 - static_cast<double>(this->m_FixedKernelBSplineOrder) / 2.0 ;
    this->m_MovingParzenTermToIndexOffset = 
      0.5 - static_cast<double>(this->m_MovingKernelBSplineOrder) / 2.0 ;
             
  } // end InitializeKernels


  /**
   * ****************** CheckForBSplineInterpolator **********************
	 */

	template <class TFixedImage, class TMovingImage> 
		void
		MattesMutualInformationImageToImageMetric3<TFixedImage,TMovingImage>
		::CheckForBSplineInterpolator(void)
  {
    /** Check if the interpolator is of type BSplineInterpolateImageFunction.
		* If so, we can make use of its EvaluateDerivatives method.
		* Otherwise, we instantiate an external central difference
		* derivative calculator. */
		this->m_InterpolatorIsBSpline = false;
		BSplineInterpolatorType * testPtr = 
      dynamic_cast<BSplineInterpolatorType *>( this->m_Interpolator.GetPointer() );
		if ( !testPtr )
    {
			this->m_DerivativeCalculator = DerivativeFunctionType::New();
			this->m_DerivativeCalculator->SetInputImage( this->m_MovingImage );
			this->m_BSplineInterpolator = 0;
			itkDebugMacro( "Interpolator is not BSpline" );
    } 
		else
    {
      this->m_InterpolatorIsBSpline = true;
			this->m_BSplineInterpolator = testPtr;
			this->m_DerivativeCalculator = 0;
			itkDebugMacro( "Interpolator is BSpline" );
    }
  } // end CheckForBSplineInterpolator

  
  /**
   * ****************** CheckForBSplineTransform **********************
   * Check if the transform is of type BSplineDeformableTransform.
	 * If so, we can speed up derivative calculations by only inspecting
	 * the parameters in the support region of a point. 
	 */

	template <class TFixedImage, class TMovingImage> 
		void
		MattesMutualInformationImageToImageMetric3<TFixedImage,TMovingImage>
		::CheckForBSplineTransform(void)
  {
		this->m_TransformIsBSpline = false;
		
		BSplineTransformType * testPtr1 = dynamic_cast<BSplineTransformType *>(
			this->m_Transform.GetPointer() );
		if( !testPtr1 )
    {
			this->m_BSplineTransform = 0;
			itkDebugMacro( "Transform is not BSplineDeformable" );
    }
		else
    {
      this->m_TransformIsBSpline = true;
			this->m_BSplineTransform = testPtr1;
			this->m_NumParametersPerDim = 
        this->m_BSplineTransform->GetNumberOfParametersPerDimension();
			this->m_NumBSplineWeights = this->m_BSplineTransform->GetNumberOfWeights();
			itkDebugMacro( "Transform is BSplineDeformable" );
    }

		/** Check if the transform is of type BSplineCombinationTransform. */
		this->m_TransformIsBSplineCombination = false;
		
		BSplineCombinationTransformType * testPtr2 = 
			dynamic_cast<BSplineCombinationTransformType *>(this->m_Transform.GetPointer() );
		if( !testPtr2 )
    {
			this->m_BSplineCombinationTransform = 0;
			itkDebugMacro( "Transform is not BSplineCombination" );
    }
		else
    {
      this->m_TransformIsBSplineCombination = true;
			this->m_BSplineCombinationTransform = testPtr2;

			/** The current transform in the BSplineCombinationTransform is 
			 * always a BSplineTransform */
			BSplineTransformType * bsplineTransform = 
				dynamic_cast<BSplineTransformType * >(
				this->m_BSplineCombinationTransform->GetCurrentTransform() );

			if (!bsplineTransform)
			{
				itkExceptionMacro(<< "The BSplineCombinationTransform is not properly configured. The CurrentTransform is not set." );
			}
			this->m_NumParametersPerDim = 
        bsplineTransform->GetNumberOfParametersPerDimension();
			this->m_NumBSplineWeights = bsplineTransform->GetNumberOfWeights();
			itkDebugMacro( "Transform is BSplineCombination" );
    }

    /** Resize the weights and transform index arrays and compute the parameters offset */
		if ( this->m_TransformIsBSpline || this->m_TransformIsBSplineCombination )
    {
			this->m_BSplineTransformWeights =
        BSplineTransformWeightsType( this->m_NumBSplineWeights );
			this->m_BSplineTransformIndices =
        BSplineTransformIndexArrayType( this->m_NumBSplineWeights );
			for ( unsigned int j = 0; j < FixedImageDimension; j++ )
      {
				this->m_ParametersOffset[j] = j * this->m_NumParametersPerDim; 
      }
      this->m_ImageJacobian.SetSize( FixedImageDimension * this->m_NumBSplineWeights );
      this->m_MaskJacobian.SetSize( FixedImageDimension * this->m_NumBSplineWeights );
      this->m_NonZeroJacobian.SetSize( FixedImageDimension * this->m_NumBSplineWeights );
    }
    else
    {   
      this->m_ImageJacobian.SetSize( this->m_NumberOfParameters );
      this->m_MaskJacobian.SetSize( this->m_NumberOfParameters );
      this->m_NonZeroJacobian.SetSize( 0 );
    }
    		
  } // end CheckForBSplineTransform


  /**
	 * ********************* InitializeInternalMasks *********************
   * Initialize the internal moving image mask
	 */

  template <class TFixedImage, class TMovingImage>
    void
    MattesMutualInformationImageToImageMetric3<TFixedImage,TMovingImage>
    ::InitializeInternalMasks(void)
  {
    typedef typename MovingImageType::RegionType                 MovingRegionType;
    typedef itk::ImageRegionExclusionIteratorWithIndex<
      InternalMovingImageMaskType>                               MovingEdgeIteratorType;
    typedef itk::ImageRegionIteratorWithIndex<
      InternalMovingImageMaskType>                               MovingIteratorType;
    typedef itk::BinaryBallStructuringElement<
      InternalMaskPixelType, MovingImageDimension >              ErosionKernelType;
    typedef itk::BinaryErodeImageFilter<
      InternalMovingImageMaskType,
      InternalMovingImageMaskType, 
      ErosionKernelType >                                        ErodeImageFilterType;
    
    /** Check if the user wants to use a differentiable overlap */
    if ( ! this->m_UseDifferentiableOverlap )
    {
      this->m_InternalMovingImageMask = 0;
      return;
    }

    /** Prepare the internal mask image */
    this->m_InternalMovingImageMask = InternalMovingImageMaskType::New();
    this->m_InternalMovingImageMask->SetRegions( 
      this->GetMovingImage()->GetLargestPossibleRegion() );
    this->m_InternalMovingImageMask->Allocate();
    this->m_InternalMovingImageMask->SetOrigin(
      this->GetMovingImage()->GetOrigin() );
    this->m_InternalMovingImageMask->SetSpacing(
      this->GetMovingImage()->GetSpacing() );

    /** Radius to erode masks */
    const unsigned int radius = this->GetMovingImageMaskInterpolationOrder();

    /** Determine inner region */
    MovingRegionType innerRegion =
      this->m_InternalMovingImageMask->GetLargestPossibleRegion();
    for (unsigned int i=0; i < MovingImageDimension; ++i)
    {
      if ( innerRegion.GetSize()[i] >= 2*radius )
      {
        /** region is large enough to crop; adjust size and index */
        innerRegion.SetSize( i, innerRegion.GetSize()[i] - 2*radius );
        innerRegion.SetIndex( i, innerRegion.GetIndex()[i] + radius );
      }
      else
      {
         innerRegion.SetSize( i, 0);
      }
    }
      
    if ( this->GetMovingImageMask() == 0 )
    {
      /** Fill the internal moving mask with ones */
      this->m_InternalMovingImageMask->FillBuffer(
        itk::NumericTraits<InternalMaskPixelType>::One );
    
      MovingEdgeIteratorType edgeIterator( this->m_InternalMovingImageMask, 
        this->m_InternalMovingImageMask->GetLargestPossibleRegion() );
      edgeIterator.SetExclusionRegion( innerRegion );
      
      /** Set the edges to zero */
      for( edgeIterator.GoToBegin(); ! edgeIterator.IsAtEnd(); ++ edgeIterator )
      {
        edgeIterator.Value() = itk::NumericTraits<InternalMaskPixelType>::Zero;
      }
      
    } // end if no moving mask
    else
    {
      /** Fill the internal moving mask with zeros */
      this->m_InternalMovingImageMask->FillBuffer(
        itk::NumericTraits<InternalMaskPixelType>::Zero );

      MovingIteratorType iterator( this->m_InternalMovingImageMask, innerRegion);
      MovingImagePointType point;

      /** Set the pixel 1 if inside the mask and to 0 if outside */
      for( iterator.GoToBegin(); ! iterator.IsAtEnd(); ++ iterator )
      {
        const MovingImageIndexType & index = iterator.GetIndex();
        this->m_InternalMovingImageMask->TransformIndexToPhysicalPoint(index, point);
        iterator.Value() = static_cast<InternalMaskPixelType>(
          this->m_MovingImageMask->IsInside(point) );
      }

      /** Erode it with a radius of 2 */
      typename InternalMovingImageMaskType::Pointer tempImage = 0;
      ErosionKernelType kernel;
      kernel.SetRadius(radius);
      kernel.CreateStructuringElement();
      typename ErodeImageFilterType::Pointer eroder = ErodeImageFilterType::New();
      eroder->SetKernel( kernel );
      eroder->SetForegroundValue( itk::NumericTraits< InternalMaskPixelType >::One  );
	    eroder->SetBackgroundValue( itk::NumericTraits< InternalMaskPixelType >::Zero );
      eroder->SetInput( this->m_InternalMovingImageMask );
      eroder->Update();
      tempImage = eroder->GetOutput();
      tempImage->DisconnectPipeline();
      this->m_InternalMovingImageMask = tempImage;
        
    } // end else (if moving mask)
        
    /** Set the internal mask into the interpolator */
    this->m_MovingImageMaskInterpolator->SetInputImage( this->m_InternalMovingImageMask );
 
  } // end InitializeInternalMasks


  /**
	 * **************** EvaluateMovingMaskValue *******************
   * Estimate value of internal moving mask 
   */

	template < class TFixedImage, class TMovingImage> 
		void
		MattesMutualInformationImageToImageMetric3<TFixedImage,TMovingImage>
    ::EvaluateMovingMaskValue(
      const MovingImagePointType & point,
      double & value) const
  {
    if ( this->m_UseDifferentiableOverlap )
    {
      /** NB: a spelling error in the itkImageFunction class! Continous... */
      MovingImageContinuousIndexType cindex;
      this->m_MovingImageMaskInterpolator->ConvertPointToContinousIndex( point, cindex);
  
      /** Compute the value of the mask */
      if ( this->m_MovingImageMaskInterpolator->IsInsideBuffer( cindex ) )
      {
        value = static_cast<double>(
          this->m_MovingImageMaskInterpolator->EvaluateAtContinuousIndex(cindex) );
      }
      else
      {
        value = 0.0;
      }
    }
    else
    {
       /** Use the original mask */
      if ( this->m_MovingImageMask.IsNotNull() )
      {
        value = static_cast<double>(
          static_cast<unsigned char>( this->m_MovingImageMask->IsInside( point ) ) );
      }
      else
      {
        value = 1.0;
      }
    }
  
  } // end EvaluateMovingMaskValue


	/**
	 * **************** EvaluateMovingMaskValueAndDerivative *******************
   * Estimate value and spatial derivative of internal moving mask 
   */

	template < class TFixedImage, class TMovingImage> 
		void
		MattesMutualInformationImageToImageMetric3<TFixedImage,TMovingImage>
    ::EvaluateMovingMaskValueAndDerivative(
      const MovingImagePointType & point,
      double & value,
      MovingImageMaskDerivativeType & derivative) const
  {
    typedef typename MovingImageMaskDerivativeType::ValueType DerivativeValueType;
       
    /** Compute the value and derivative of the mask */

    if ( this->m_UseDifferentiableOverlap )
    {
      /** NB: a spelling error in the itkImageFunction class! Continous... */
      MovingImageContinuousIndexType cindex;
      this->m_MovingImageMaskInterpolator->ConvertPointToContinousIndex( point, cindex);

      if ( this->m_MovingImageMaskInterpolator->IsInsideBuffer( cindex ) )
      {
        value = static_cast<double>(
          this->m_MovingImageMaskInterpolator->EvaluateAtContinuousIndex(cindex) );
        derivative = 
          this->m_MovingImageMaskInterpolator->EvaluateDerivativeAtContinuousIndex(cindex);
      }
      else
      {
        value = 0.0;
        derivative.Fill( itk::NumericTraits<DerivativeValueType>::Zero );
      }
    }
    else
    {
      /** Just ignore the derivative of the mask */
      if ( this->m_MovingImageMask.IsNotNull() )
      {
        value = static_cast<double>(
          static_cast<unsigned char>( this->m_MovingImageMask->IsInside( point ) ) );
      }
      else
      {
        value = 1.0;
      }
      derivative.Fill( itk::NumericTraits<DerivativeValueType>::Zero );
    }
  } // end EvaluateMovingMaskValueAndDerivative


  /**
	 * ************************** GetValue **************************
	 * Get the match Measure.
	 */

	template < class TFixedImage, class TMovingImage  >
	  typename MattesMutualInformationImageToImageMetric3<TFixedImage,TMovingImage>
	  ::MeasureType
	  MattesMutualInformationImageToImageMetric3<TFixedImage,TMovingImage>
	  ::GetValue( const ParametersType& parameters ) const
	{		 
    /** Reset PDFs to all zeros.
    * Assumed the size has already been set to NumberOfHistogramBins in Initialize().*/
    this->m_FixedImageMarginalPDF.Fill(0.0);
    this->m_MovingImageMarginalPDF.Fill(0.0);
    this->m_JointPDF->FillBuffer( 0.0 );
    
    // Set up the parameters in the transform
    this->m_Transform->SetParameters( parameters );

    /** Update the imageSampler and get a handle to the sample container. */
    this->GetImageSampler()->Update();
    ImageSampleContainerPointer sampleContainer = this->GetImageSampler()->GetOutput();

    /** Create iterator over the sample container. */
    typename ImageSampleContainerType::ConstIterator fiter;
    typename ImageSampleContainerType::ConstIterator fbegin = sampleContainer->Begin();
    typename ImageSampleContainerType::ConstIterator fend = sampleContainer->End();

    unsigned long nSamples=0;
    unsigned long nFixedImageSamples=0;
    double sumOfMovingMaskValues = 0.0;
        
    for ( fiter = fbegin; fiter != fend; ++fiter )
    {       
      ++nFixedImageSamples;
      
      /** Read image values and coordinates and initialize some variables */
      double fixedImageValue = static_cast<double>( (*fiter).Value().m_ImageValue );
      const FixedImagePointType & fixedPoint = (*fiter).Value().m_ImageCoordinates;
      double movingImageValue; 
      MovingImagePointType mappedPoint;
      bool sampleOk;
            
      /** Transform point and check if it is inside the bspline support region */
      this->TransformPoint( fixedPoint, mappedPoint, sampleOk);

      /** Check if point is inside mask */
      double movingMaskValue = 0.0;
      if ( sampleOk ) 
      {
        this->EvaluateMovingMaskValue( mappedPoint, movingMaskValue );
        const double smallNumber1 = 1e-10;
        sampleOk = movingMaskValue > smallNumber1;
      }

      /** Compute the moving image value and check if the point is
      * inside the moving image buffer */
      if ( sampleOk )
      {
        this->EvaluateMovingImageValueAndDerivative(
          mappedPoint, sampleOk, movingImageValue, 0 );
      }
      
      if( sampleOk )
      {
        ++nSamples; 
        sumOfMovingMaskValues += movingMaskValue;

        /** Make sure the fixed image value falls within the histogram range */
        fixedImageValue = vnl_math_min( fixedImageValue, this->m_FixedImageMaxLimit );
        fixedImageValue = vnl_math_max( fixedImageValue, this->m_FixedImageMinLimit );
    
        /** Compute this sample's contribution to the marginal and joint distributions. */
        this->UpdateJointPDFAndDerivatives( 
          fixedImageValue, movingImageValue, movingMaskValue, false);
      }       

    } // end iterating over fixed image spatial sample container for loop
    
    this->CheckNumberOfSamples(
      sampleContainer->Size(), nSamples, sumOfMovingMaskValues);

    this->NormalizeJointPDF( this->m_JointPDF, 1.0 / sumOfMovingMaskValues );

    this->ComputeMarginalPDF( this->m_JointPDF, this->m_FixedImageMarginalPDF, 0 );
    this->ComputeMarginalPDF( this->m_JointPDF, this->m_MovingImageMarginalPDF, 1 );
          
    /** Compute the metric by double summation over histogram. */

    /** Setup pointer to point to the first bin */
    JointPDFValueType * jointPDFPtr = this->m_JointPDF->GetBufferPointer();

    /** Loop over histogram */
    double sum = 0.0;
    for( unsigned int fixedIndex = 0; fixedIndex < this->m_NumberOfFixedHistogramBins; ++fixedIndex )
    {
      const double fixedImagePDFValue = this->m_FixedImageMarginalPDF[fixedIndex];
      for( unsigned int movingIndex = 0; movingIndex < this->m_NumberOfMovingHistogramBins;
        ++movingIndex, jointPDFPtr++ )      
      {
        const double movingImagePDFValue = this->m_MovingImageMarginalPDF[movingIndex];
        const double fixPDFmovPDF = fixedImagePDFValue * movingImagePDFValue;
        const double jointPDFValue = *(jointPDFPtr);
                
        /** check for non-zero bin contribution */
        if( jointPDFValue > 1e-16 && fixPDFmovPDF > 1e-16 )
        {
          sum += jointPDFValue *
            vcl_log( jointPDFValue / fixPDFmovPDF );

        }  // end if-block to check non-zero bin contribution
      }  // end for-loop over moving index
    }  // end for-loop over fixed index
    
    return static_cast<MeasureType>( -1.0 * sum );
    
  } // end GetValue


	/**
	 * ******************** GetValueAndDerivative *******************
	 * Get both the Value and the Derivative of the Measure. 
	 */

	template < class TFixedImage, class TMovingImage  >
	  void
	  MattesMutualInformationImageToImageMetric3<TFixedImage,TMovingImage>
	  ::GetValueAndDerivative(
	  const ParametersType& parameters,
	  MeasureType& value,
	  DerivativeType& derivative) const
	{		 
    /**  Set output values to zero */
    value = NumericTraits< MeasureType >::Zero;
    derivative = DerivativeType( this->GetNumberOfParameters() );
    derivative.Fill( NumericTraits< MeasureType >::Zero );
      
    /** Reset PDFs to all zeros.
    * Assumed the size has already been set to NumberOfHistogramBins in Initialize().*/
    this->m_FixedImageMarginalPDF.Fill(0.0);
    this->m_MovingImageMarginalPDF.Fill(0.0);
    this->m_JointPDF->FillBuffer( 0.0 );
    this->m_JointPDFDerivatives->FillBuffer( 0.0 );

    /** Reset the AlphaDerivative */
    this->m_AlphaDerivatives.Fill(0.0);
        
    /** Set up the parameters in the transform */
    this->m_Transform->SetParameters( parameters );

    /** Update the imageSampler and get a handle to the sample container. */
    this->GetImageSampler()->Update();
    ImageSampleContainerPointer sampleContainer = this->GetImageSampler()->GetOutput();

    /** Create iterator over the sample container. */
    typename ImageSampleContainerType::ConstIterator fiter;
    typename ImageSampleContainerType::ConstIterator fbegin = sampleContainer->Begin();
    typename ImageSampleContainerType::ConstIterator fend = sampleContainer->End();

    unsigned long nSamples=0;
    unsigned long nFixedImageSamples=0;
    double sumOfMovingMaskValues = 0.0;
    
    for ( fiter = fbegin; fiter != fend; ++fiter )
    {
      ++nFixedImageSamples;

      /** Read image values and coordinates and initialize some variables */
      double fixedImageValue = static_cast<double>( (*fiter).Value().m_ImageValue );
      const FixedImagePointType & fixedPoint = (*fiter).Value().m_ImageCoordinates;
      double movingImageValue; 
      MovingImagePointType mappedPoint;
      bool sampleOk;
      ImageDerivativesType movingImageGradientValue;
            
      /** Transform point and check if it is inside the bspline support region */
      this->TransformPoint( fixedPoint, mappedPoint, sampleOk);
      
      /** Check if point is inside mask */
      double movingMaskValue = 0.0;
      MovingImageMaskDerivativeType movingMaskDerivative; 
      if ( sampleOk ) 
      {
        this->EvaluateMovingMaskValueAndDerivative(
          mappedPoint, movingMaskValue, movingMaskDerivative );
        const double movingMaskDerivativeMagnitude = movingMaskDerivative.GetNorm();
        const double smallNumber1 = 1e-10;
        sampleOk = ( movingMaskValue > smallNumber1 ) ||
          ( movingMaskDerivativeMagnitude > smallNumber1 );
      }
    
      /** Compute the moving image value and derivative and check if the point is
      * inside the moving image buffer */
      if ( sampleOk )
      {
        this->EvaluateMovingImageValueAndDerivative( 
          mappedPoint, sampleOk, movingImageValue, &movingImageGradientValue );
      }
            
      if( sampleOk )
      {
        ++nSamples; 
        sumOfMovingMaskValues += movingMaskValue;

        /** Make sure the fixed image value falls within the histogram range */
        fixedImageValue = vnl_math_min( fixedImageValue, this->m_FixedImageMaxLimit );
        fixedImageValue = vnl_math_max( fixedImageValue, this->m_FixedImageMinLimit );
            
        this->ComputeTransformJacobianInnerProducts( 
          fixedPoint, movingImageGradientValue, movingMaskDerivative );

        this->UpdateAlphaDerivatives();
        
        this->UpdateJointPDFAndDerivatives(
          fixedImageValue, movingImageValue, movingMaskValue, true );
                              
      } //end if-block check sampleOk
    } // end iterating over fixed image spatial sample container for loop
    
    this->CheckNumberOfSamples(
      sampleContainer->Size(), nSamples, sumOfMovingMaskValues);

    this->NormalizeJointPDF(
      this->m_JointPDF, 1.0 / sumOfMovingMaskValues );
    this->NormalizeJointPDFDerivatives(
      this->m_JointPDFDerivatives, 1.0 / sumOfMovingMaskValues );

    this->ComputeMarginalPDF( this->m_JointPDF, this->m_FixedImageMarginalPDF, 0 );
    this->ComputeMarginalPDF( this->m_JointPDF, this->m_MovingImageMarginalPDF, 1 );
       
    /** Compute the metric by double summation over histogram. */

    /**  Setup pointer to point to the first bin */
    JointPDFValueType * jointPDFPtr = m_JointPDF->GetBufferPointer();

    /** Initialize sum to zero */
    double sum = 0.0;
    typedef typename JointPDFDerivativesType::OffsetValueType PDFDerivativesOffsetValueType;
    const PDFDerivativesOffsetValueType offset1 =
      this->m_JointPDFDerivatives->GetOffsetTable()[1];
    const PDFDerivativesOffsetValueType offset2 =
      this->m_JointPDFDerivatives->GetOffsetTable()[2];

    for( unsigned int fixedIndex = 0; fixedIndex < this->m_NumberOfFixedHistogramBins;
      ++fixedIndex )
    {
      const double fixedImagePDFValue = this->m_FixedImageMarginalPDF[fixedIndex];
      for( unsigned int movingIndex = 0; movingIndex < this->m_NumberOfMovingHistogramBins; 
        ++movingIndex, jointPDFPtr++ )      
      {
        const double movingImagePDFValue = this->m_MovingImageMarginalPDF[movingIndex];
        const double fixPDFmovPDF = fixedImagePDFValue * movingImagePDFValue;
        const double jointPDFValue = *(jointPDFPtr);
        /** check for non-zero bin contribution */
        if( jointPDFValue > 1e-16 && fixPDFmovPDF > 1e-16 )
        {
          const double pRatio = vcl_log( jointPDFValue / fixPDFmovPDF );
          sum += jointPDFValue * pRatio;
        
          /** move joint pdf derivative pointer to the right position */
          JointPDFValueType * derivPtr =
            this->m_JointPDFDerivatives->GetBufferPointer() +
            ( fixedIndex * offset2 ) +
            ( movingIndex * offset1 );

          for( unsigned int parameter=0; parameter < this->m_NumberOfParameters;
              ++parameter, derivPtr++ )
          {                    
            /**  Ref: eqn 23 of Thevenaz & Unser paper [3] */
            derivative[parameter] -= (*derivPtr) * pRatio;
                        
          }  // end for-loop over parameters
        }  // end if-block to check non-zero bin contribution
      }  // end for-loop over moving index
    }  // end for-loop over fixed index
    
    value = static_cast<MeasureType>( -1.0 * sum );

    /** Add -da/dmu sum_i sum_k (h log h / a hT hR)
     *  
     * note: alphaDerivative should still be multiplied by -alpha^2/e_T e_R
     * in order to make it the real derivative of alpha.
     * this is done implicitly in the following equations.
     */
    const double alphaDerivativeFactor = sum / sumOfMovingMaskValues;
    for( unsigned int parameter=0; parameter < this->m_NumberOfParameters; ++parameter)
    {
      derivative[parameter] += alphaDerivativeFactor * this->m_AlphaDerivatives[parameter];
    }
    
  } // end GetValueAndDerivative


	/**
	 * ******************** GetDerivative ***************************
	 *
	 * Get the match measure derivative.
	 */

	template < class TFixedImage, class TMovingImage  >
	void
	MattesMutualInformationImageToImageMetric3<TFixedImage,TMovingImage>
	::GetDerivative( const ParametersType& parameters, DerivativeType & derivative ) const
	{
		MeasureType value;
		// call the combined version
		this->GetValueAndDerivative( parameters, value, derivative );

	} // end GetDerivative


	/**
	 * ****************** ComputeImageDerivatives *******************
	 *
	 * Compute image derivatives using a central difference function
	 * if we are not using a BSplineInterpolator, which includes
	 * derivatives. 
   * This function is called by EvaluateMovingImageValueAndDerivative 
   */

	template < class TFixedImage, class TMovingImage >
		void
		MattesMutualInformationImageToImageMetric3<TFixedImage,TMovingImage>
		::ComputeImageDerivatives( 
		const MovingImageContinuousIndexType & cindex, 
		ImageDerivativesType& gradient ) const
	{		
		if( this->m_InterpolatorIsBSpline )
		{
			// Computed moving image gradient using derivative BSpline kernel.
			gradient = 
        this->m_BSplineInterpolator->EvaluateDerivativeAtContinuousIndex( cindex );
		}
		else
		{
			// For all generic interpolator use central differencing.
			gradient =
        this->m_DerivativeCalculator->EvaluateAtContinuousIndex( cindex );
		}
		
	} // end ComputeImageDerivatives
	

	/**
	 * ********************** TransformPoint ************************
	 *
	 * Transform a point from FixedImage domain to MovingImage domain.
	 * This function also checks if mapped point is within support region
   * and mask.
	 */

	template < class TFixedImage, class TMovingImage >
		void
		MattesMutualInformationImageToImageMetric3<TFixedImage,TMovingImage>
		::TransformPoint( 
		const FixedImagePointType& fixedImagePoint, 
		MovingImagePointType& mappedPoint,
		bool& sampleOk	) const
	{
		if ( !(this->m_TransformIsBSpline) && !(this->m_TransformIsBSplineCombination) )
		{
			mappedPoint = this->m_Transform->TransformPoint( fixedImagePoint );
      sampleOk = true;
		}
		else
		{
			if (this->m_TransformIsBSpline)
			{
				this->m_BSplineTransform->TransformPoint( 
          fixedImagePoint,
					mappedPoint,
					this->m_BSplineTransformWeights,
					this->m_BSplineTransformIndices,
					sampleOk );
			}
			else if (this->m_TransformIsBSplineCombination)
			{
				this->m_BSplineCombinationTransform->TransformPoint( 
          fixedImagePoint,
				  mappedPoint,
				  this->m_BSplineTransformWeights,
				  this->m_BSplineTransformIndices,
				  sampleOk );
			}
		}
	} // end TransformPoint

  	
	/**
	 * ******************* EvaluateMovingImageValueAndDerivative ******************
	 *
	 * Compute image value and possibly derivative at a transformed point
	 */

  template < class TFixedImage, class TMovingImage >
		void
		MattesMutualInformationImageToImageMetric3<TFixedImage,TMovingImage>
    ::EvaluateMovingImageValueAndDerivative( 
    const MovingImagePointType & mappedPoint,
    bool & sampleOk,
    double & movingImageValue,
    ImageDerivativesType * gradient) const
  {
    // Check if mapped point inside image buffer
    MovingImageContinuousIndexType cindex;
    this->m_Interpolator->ConvertPointToContinousIndex( mappedPoint, cindex);
    
		sampleOk = this->m_Interpolator->IsInsideBuffer( cindex );
		
		if ( sampleOk )
    {
      /** Compute value and possibly derivative */
      movingImageValue = this->m_Interpolator->EvaluateAtContinuousIndex( cindex );
      if ( gradient )
      {    
        this->ComputeImageDerivatives( cindex, *gradient);
      }
      
      if ( this->m_SoftLimitMovingGrayValues )
      {
        /** Apply a soft limit */
        const double diff =  movingImageValue - this->m_MovingImageTrueMax;
        if ( diff > 1e-10 )
        {
          const double temp = this->m_SoftMaxLimit_A *
            vcl_exp( this->m_SoftMaxLimit_a * diff );
          movingImageValue = temp + this->m_MovingImageMaxLimit;
          if (gradient)
          {
            const double gradientfactor = this->m_SoftMaxLimit_a * temp;
            for (unsigned int i = 0; i < MovingImageDimension; ++i)
            {
              (*gradient)[i] = (*gradient)[i] * gradientfactor;
            }
          }
        }
        else
        {
          const double diff = movingImageValue - this->m_MovingImageTrueMin;
          if ( diff < -1e-10 )
          {
            const double temp = this->m_SoftMinLimit_A * vcl_exp( 
              this->m_SoftMinLimit_a * ( movingImageValue - this->m_MovingImageTrueMin )  );
            movingImageValue = temp + this->m_MovingImageMinLimit;
            if (gradient)
            {
              const double gradientfactor = this->m_SoftMinLimit_a * temp;
              for (unsigned int i = 0; i < MovingImageDimension; ++i)
              {
                (*gradient)[i] = (*gradient)[i] * gradientfactor;
              } // end for
            } // end if gradient
          } // end if diff < -1e10            
        } // end else
      } //end if softlimit
      else if ( this->m_HardLimitMovingGrayValues )
			{ 
        /** Limit the image value to the image's maximum and minimum */
        movingImageValue = vnl_math_min( movingImageValue, this->m_MovingImageMaxLimit );
        movingImageValue = vnl_math_max( movingImageValue, this->m_MovingImageMinLimit );
        if ( gradient )
        {
          (*gradient).Fill(0.0); 
        }
      }
      else
      { 
        /** Throw out the sample */
        sampleOk = ! ( (movingImageValue < this->m_MovingImageTrueMin) ||
          ( movingImageValue > this->m_MovingImageTrueMax ) ); 
      }
    }

	} // end EvaluateMovingImageValueAndDerivative


  /*
   * ********************** ComputeParzenValues ***************
   */

  template < class TFixedImage, class TMovingImage >
		void
		MattesMutualInformationImageToImageMetric3<TFixedImage,TMovingImage>
		::ComputeParzenValues(
      double parzenWindowTerm, int parzenWindowIndex,
      const KernelFunctionType * kernel, ParzenValueContainerType & parzenValues) const
  {
    const unsigned int max_i = parzenValues.GetSize();
    for ( unsigned int i = 0 ; i < max_i; ++i, ++parzenWindowIndex )
    {
      parzenValues[i] = kernel->Evaluate( 
        static_cast<double>(parzenWindowIndex) - parzenWindowTerm );        
    }
  } // end ComputeParzenValues


  /**
   * ********************** UpdateJointPDF ***************
   */

  template < class TFixedImage, class TMovingImage >
		void
		MattesMutualInformationImageToImageMetric3<TFixedImage,TMovingImage>
		::UpdateJointPDFAndDerivatives(
      double fixedImageValue, double movingImageValue, double movingMaskValue, 
      bool updateDerivatives) const
  {
    typedef ImageSliceIteratorWithIndex< JointPDFType >  PDFIteratorType;

    /** Determine parzen window arguments (see eqn 6 of Mattes paper [2]). */
    const double fixedImageParzenWindowTerm = 
      fixedImageValue / this->m_FixedImageBinSize - this->m_FixedImageNormalizedMin;
		const double movingImageParzenWindowTerm =
	    movingImageValue / this->m_MovingImageBinSize - this->m_MovingImageNormalizedMin;
        
    /** The lowest bin numbers affected by this pixel: */
    const int fixedImageParzenWindowIndex = 
		  static_cast<int>( vcl_floor( 
      fixedImageParzenWindowTerm + this->m_FixedParzenTermToIndexOffset ) );
    const int movingImageParzenWindowIndex =		
		  static_cast<int>( vcl_floor(
      movingImageParzenWindowTerm + this->m_MovingParzenTermToIndexOffset ) );

    /** The parzen values */
    ParzenValueContainerType fixedParzenValues( this->m_JointPDFWindow.GetSize()[1] );
    ParzenValueContainerType movingParzenValues( this->m_JointPDFWindow.GetSize()[0] );
    this->ComputeParzenValues(
      fixedImageParzenWindowTerm, fixedImageParzenWindowIndex,
      this->m_FixedKernel, fixedParzenValues);
    this->ComputeParzenValues(
      movingImageParzenWindowTerm, movingImageParzenWindowIndex,
      this->m_MovingKernel, movingParzenValues);
    
    /** Position the JointPDFWindow */
    JointPDFIndexType pdfWindowIndex;
    pdfWindowIndex[0] = movingImageParzenWindowIndex;
    pdfWindowIndex[1] = fixedImageParzenWindowIndex;
    this->m_JointPDFWindow.SetIndex( pdfWindowIndex );

    PDFIteratorType it( this->m_JointPDF, this->m_JointPDFWindow );
    it.GoToBegin();
    it.SetFirstDirection(0);
    it.SetSecondDirection(1);
    
    if ( !updateDerivatives )
    {  
      /** Loop over the parzen window region and increment the values */    
      for ( unsigned int f = 0; f < fixedParzenValues.GetSize(); ++f )
      {
        const double fv_mask = fixedParzenValues[f] * movingMaskValue;
        for ( unsigned int m = 0; m < movingParzenValues.GetSize(); ++m )
        {
          it.Value() += static_cast<PDFValueType>( fv_mask * movingParzenValues[m] );
          ++it;
        }
        it.NextLine();
      }
    }
    else
    {
      /** Compute the derivatives of the moving parzen window */
      ParzenValueContainerType derivativeMovingParzenValues(
        this->m_JointPDFWindow.GetSize()[0] );
      this->ComputeParzenValues(
        movingImageParzenWindowTerm, movingImageParzenWindowIndex,
        this->m_DerivativeMovingKernel, derivativeMovingParzenValues);

      const double mask_et = movingMaskValue / this->m_MovingImageBinSize;

      /** Loop over the parzen window region and increment the values
       * Also update the pdf derivatives */    
      for ( unsigned int f = 0; f < fixedParzenValues.GetSize(); ++f )
      {
        const double fv = fixedParzenValues[f];
        const double fv_mask_et = fv * mask_et;
        for ( unsigned int m = 0; m < movingParzenValues.GetSize(); ++m )
        {
          const double fv_mv = fv * movingParzenValues[m];
          it.Value() += static_cast<PDFValueType>( fv_mv * movingMaskValue );
          this->UpdateJointPDFDerivatives( 
            it.GetIndex(), fv_mask_et * derivativeMovingParzenValues[m], fv_mv);
          ++it;
        }
        it.NextLine();
      }
    }

  } // end UpdateJointPDF
  
 
	/**
	 * *************** ComputeTransformJacobianInnerProducts ****************
	 */

	template < class TFixedImage, class TMovingImage >
		void
		MattesMutualInformationImageToImageMetric3<TFixedImage,TMovingImage>
		::ComputeTransformJacobianInnerProducts( 
		const FixedImagePointType & fixedImagePoint, 
		const ImageDerivativesType & movingImageGradientValue,
    const MovingImageMaskDerivativeType & movingMaskDerivative) const
	{
		if( !(this->m_TransformIsBSpline) && !(this->m_TransformIsBSplineCombination) )
		{
			/** Generic version which works for all transforms. */
			
			/** Compute the transform Jacobian. */
			typedef typename TransformType::JacobianType JacobianType;
			const JacobianType& jacobian = 
				this->m_Transform->GetJacobian( fixedImagePoint );
			
			for ( unsigned int mu = 0; mu < m_NumberOfParameters; mu++ )
			{
				double imjac = 0.0;
        double maskjac = 0.0;
				for ( unsigned int dim = 0; dim < FixedImageDimension; dim++ )
				{
					imjac   += jacobian[dim][mu] * movingImageGradientValue[dim];
          maskjac += jacobian[dim][mu] * movingMaskDerivative[dim];          
				} //end dim loop

        this->m_ImageJacobian[mu] = imjac;
        this->m_MaskJacobian[mu]  = maskjac;
        				
			} //end mu loop
		} // end if no bspline transform
		else
		{
			/** If the transform is of type BSplineDeformableTransform or of type
			 * BSplineCombinationTransform, we can obtain a speed up by only 
			 * processing the affected parameters. */
      unsigned int i = 0;
			for( unsigned int dim = 0; dim < FixedImageDimension; dim++ )
			{
				for( unsigned int mu = 0; mu < this->m_NumBSplineWeights; mu++ )
				{
				  /* The array weights contains the Jacobian values in a 1-D array 
					 * (because for each parameter the Jacobian is non-zero in only 1 of the
					 * possible dimensions) which is multiplied by the moving image gradient. */
					const double imjac   = 
            movingImageGradientValue[dim] * this->m_BSplineTransformWeights[mu];
          const double maskjac = 
            movingMaskDerivative[dim] * this->m_BSplineTransformWeights[mu];
					const unsigned int parameterNumber = 
            this->m_BSplineTransformIndices[mu] + this->m_ParametersOffset[dim];

          this->m_ImageJacobian[i]   = imjac;
          this->m_MaskJacobian[i]    = maskjac;
          this->m_NonZeroJacobian[i] = parameterNumber;
          ++i;
  			} //end mu for loop
			} //end dim for loop
		} // end if-block transform is BSpline
		
	} // end ComputeTransformJacobianInnerProducts


  /**
   * *********************** NormalizeJointPDF ***********************
   * Multiply the pdf entries by the given normalization factor 
   */

  template < class TFixedImage, class TMovingImage >
		void
		MattesMutualInformationImageToImageMetric3<TFixedImage,TMovingImage>
    ::NormalizeJointPDF( JointPDFType * pdf, double factor ) const
  {
    /** Normalize the PDFs */
		typedef ImageRegionIterator<JointPDFType> JointPDFIteratorType;
		JointPDFIteratorType it( pdf, pdf->GetBufferedRegion() );
		it.GoToBegin();
    const PDFValueType castfac = static_cast<PDFValueType>(factor);
    while( !it.IsAtEnd() )
		{
		  it.Value() *= castfac;
      ++it;
		}
	} // end NormalizeJointPDF


  /**
   * *********************** NormalizeJointPDFDerivatives ***********************
   * Multiply the pdf derivatives entries by the given normalization factor 
   */

  template < class TFixedImage, class TMovingImage >
		void
		MattesMutualInformationImageToImageMetric3<TFixedImage,TMovingImage>
    ::NormalizeJointPDFDerivatives( JointPDFDerivativesType * pdf, double factor ) const
  {
    /** Normalize the PDFs */
		typedef ImageRegionIterator<JointPDFDerivativesType> JointPDFDerivativesIteratorType;
		JointPDFDerivativesIteratorType it( pdf, pdf->GetBufferedRegion() );
		it.GoToBegin();
    const PDFValueType castfac = static_cast<PDFValueType>(factor);
    while( !it.IsAtEnd() )
		{
		  it.Value() *= castfac;
      ++it;
		}
	} // end NormalizeJointPDF
	

  /**
   * ************************ ComputeMarginalPDF ***********************
   * Compute marginal pdf by summing over the joint pdf
   * direction = 0: fixed marginal pdf
   * direction = 1: moving marginal pdf
   */

  template < class TFixedImage, class TMovingImage >
		void
		MattesMutualInformationImageToImageMetric3<TFixedImage,TMovingImage>
    ::ComputeMarginalPDF( 
      const JointPDFType * jointPDF,
      MarginalPDFType & marginalPDF, unsigned int direction ) const
  {
    typedef ImageLinearIteratorWithIndex<JointPDFType> JointPDFLinearIterator;
    JointPDFLinearIterator linearIter( 
		  this->m_JointPDF, this->m_JointPDF->GetBufferedRegion() );
    linearIter.SetDirection( direction );
		linearIter.GoToBegin();
    unsigned int marginalIndex = 0;
		while( !linearIter.IsAtEnd() )
		{
		  double sum = 0.0;
		  while( !linearIter.IsAtEndOfLine() )
		  {
	  	  sum += linearIter.Get();
 	  	  ++linearIter; 
		  }
		  marginalPDF[ marginalIndex ] = static_cast<PDFValueType>( sum );
		  linearIter.NextLine();
		  ++marginalIndex;			 
		} 
  } // end ComputeMarginalPDFs


  /**
   * *********************** CheckNumberOfSamples ***********************
   */

  template < class TFixedImage, class TMovingImage >
		void
		MattesMutualInformationImageToImageMetric3<TFixedImage,TMovingImage>
    ::CheckNumberOfSamples(
      unsigned long wanted, unsigned long found, double sumOfMaskValues) const
  {
    const double smallNumber2 = 1e-10;
    if ( this->GetCheckNumberOfSamples() || sumOfMaskValues < smallNumber2 ) 
    {
      if( found < wanted / 4 || sumOfMaskValues < smallNumber2 )
      {
        itkExceptionMacro( "Too many samples map outside moving image buffer: "
          << found << " / " << wanted << std::endl );
      }
    }
    this->m_NumberOfPixelsCounted = found;
  } // end CheckNumberOfSamples


  /**
	 * *************** UpdateJointPDFDerivatives ***************************
	 */

	template < class TFixedImage, class TMovingImage >
		void
		MattesMutualInformationImageToImageMetric3<TFixedImage,TMovingImage>
		::UpdateJointPDFDerivatives(
    const JointPDFIndexType & pdfIndex, double factor_a, double factor_b) const
	{
		/** Get the pointer to the element with index [0, pdfIndex[0], pdfIndex[1]]*/
		JointPDFValueType * derivPtr = this->m_JointPDFDerivatives->GetBufferPointer() +
      ( pdfIndex[0] * this->m_JointPDFDerivatives->GetOffsetTable()[1] ) +
			( pdfIndex[1] * this->m_JointPDFDerivatives->GetOffsetTable()[2] );
    
		if( this->m_NonZeroJacobian.GetSize() == 0 )
		{
			/** Loop over all jacobians */
			for ( unsigned int mu = 0; mu < m_NumberOfParameters; mu++, derivPtr++ )
			{
				*(derivPtr) += static_cast<PDFValueType>(
          this->m_MaskJacobian[mu] * factor_b - this->m_ImageJacobian[mu] * factor_a );
			}
		} 
		else
		{
			/** Loop only over the non-zero jacobians */
			for ( unsigned int i = 0; i < this->m_MaskJacobian.GetSize(); ++i)
      {
				const unsigned int mu = this->m_NonZeroJacobian[i];
        JointPDFValueType * ptr = derivPtr + mu;
				*(ptr) += static_cast<PDFValueType>(
          this->m_MaskJacobian[mu] * factor_b - this->m_ImageJacobian[mu] * factor_a );
			}
		}
		
	} // end UpdateJointPDFDerivatives


  /**
	 * *************** UpdateAlphaDerivatives ***************************
	 */

	template < class TFixedImage, class TMovingImage >
		void
		MattesMutualInformationImageToImageMetric3<TFixedImage,TMovingImage>
		::UpdateAlphaDerivatives( void ) const
  {
    if ( this->m_NonZeroJacobian.GetSize() == 0 )
    {
      /** Loop over all jacobians */
      for ( unsigned int i = 0; i < this->m_MaskJacobian.GetSize(); ++i)
      {
        this->m_AlphaDerivatives[i] += this->m_MaskJacobian[i]; ///use iterator?
      }
    }
    else
    {
      /** Only pick the nonzero jacobians */
      for ( unsigned int i = 0; i < this->m_MaskJacobian.GetSize(); ++i)
      {
        this->m_AlphaDerivatives[ this->m_NonZeroJacobian[i] ] += this->m_MaskJacobian[i];
      }
    }
  } // end UpdateAlphaDerivatives

} // end namespace itk 


#endif // end #ifndef _itkMattesMutualInformationImageToImageMetric3_HXX__

