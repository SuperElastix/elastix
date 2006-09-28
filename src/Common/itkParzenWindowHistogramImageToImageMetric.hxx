#ifndef _itkParzenWindowHistogramImageToImageMetric_HXX__
#define _itkParzenWindowHistogramImageToImageMetric_HXX__

#include "itkParzenWindowHistogramImageToImageMetric.h"

#include "itkImageRegionIterator.h"
#include "itkImageLinearIteratorWithIndex.h"
#include "itkImageSliceIteratorWithIndex.h"
#include "vnl/vnl_math.h"

namespace itk
{	
	
	/**
	 * ********************* Constructor ****************************
	 */

	template < class TFixedImage, class TMovingImage >
		ParzenWindowHistogramImageToImageMetric<TFixedImage,TMovingImage>
		::ParzenWindowHistogramImageToImageMetric()
	{
		this->m_NumberOfFixedHistogramBins = 50;
    this->m_NumberOfMovingHistogramBins = 50;
		this->m_JointPDF = 0;
		this->m_JointPDFDerivatives = 0;
    this->m_FixedImageNormalizedMin = 0.0;
		this->m_MovingImageNormalizedMin = 0.0;
		this->m_FixedImageBinSize = 0.0;
		this->m_MovingImageBinSize = 0.0;
    this->m_Alpha = 0.0;

    this->m_FixedKernel = 0;
    this->m_MovingKernel = 0;
		this->m_DerivativeMovingKernel = 0;
    this->m_FixedKernelBSplineOrder = 0;
    this->m_MovingKernelBSplineOrder = 3;
    this->m_FixedParzenTermToIndexOffset = 0.5;
    this->m_MovingParzenTermToIndexOffset = -1.0;
           
    this->SetUseImageSampler(true);
    this->SetUseFixedImageLimiter(true);
    this->SetUseMovingImageLimiter(true);

	} // end Constructor


	/**
	 * ********************* PrintSelf ******************************
	 *
	 * Print out internal information about this class.
	 */

	template < class TFixedImage, class TMovingImage  >
		void
		ParzenWindowHistogramImageToImageMetric<TFixedImage,TMovingImage>
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
		ParzenWindowHistogramImageToImageMetric<TFixedImage,TMovingImage>
		::Initialize(void) throw ( ExceptionObject )
	{
		/** Call the superclass to check that standard components are available */
		this->Superclass::Initialize();

    /** Set up the histograms */
    this->InitializeHistograms();

    /** Set up the Parzen windows */
    this->InitializeKernels();
    
    /** Allocate memory for the alpha derivatives.
     * Assume the superclass has set the m_NumberOfParameters */
    this->m_AlphaDerivatives.SetSize( this->m_NumberOfParameters );
         
	} // end Initialize


  /**
   * ****************** InitializeHistograms *****************************
   */

	template <class TFixedImage, class TMovingImage> 
		void
		ParzenWindowHistogramImageToImageMetric<TFixedImage,TMovingImage>
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
		ParzenWindowHistogramImageToImageMetric<TFixedImage,TMovingImage>
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
	 * ******************** GetDerivative ***************************
	 *
	 * Get the match measure derivative.
	 */

	template < class TFixedImage, class TMovingImage  >
	void
	ParzenWindowHistogramImageToImageMetric<TFixedImage,TMovingImage>
	::GetDerivative( const ParametersType& parameters, DerivativeType & derivative ) const
	{
		MeasureType value;
		// call the combined version
		this->GetValueAndDerivative( parameters, value, derivative );

	} // end GetDerivative


  /*
   * ********************** EvaluateParzenValues ***************
   */

  template < class TFixedImage, class TMovingImage >
		void
		ParzenWindowHistogramImageToImageMetric<TFixedImage,TMovingImage>
		::EvaluateParzenValues(
      double parzenWindowTerm, int parzenWindowIndex,
      const KernelFunctionType * kernel, ParzenValueContainerType & parzenValues) const
  {
    const unsigned int max_i = parzenValues.GetSize();
    for ( unsigned int i = 0 ; i < max_i; ++i, ++parzenWindowIndex )
    {
      parzenValues[i] = kernel->Evaluate( 
        static_cast<double>(parzenWindowIndex) - parzenWindowTerm );        
    }
  } // end EvaluateParzenValues


  /**
   * ********************** UpdateJointPDFAndDerivatives ***************
   */

  template < class TFixedImage, class TMovingImage >
		void
		ParzenWindowHistogramImageToImageMetric<TFixedImage,TMovingImage>
		::UpdateJointPDFAndDerivatives(
      RealType fixedImageValue, RealType movingImageValue, RealType movingMaskValue, 
      const DerivativeType * imageJacobian, const DerivativeType * maskJacobian) const
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
    this->EvaluateParzenValues(
      fixedImageParzenWindowTerm, fixedImageParzenWindowIndex,
      this->m_FixedKernel, fixedParzenValues);
    this->EvaluateParzenValues(
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
    
    if ( !imageJacobian || !maskJacobian )
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
      this->EvaluateParzenValues(
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
            it.GetIndex(), fv_mask_et * derivativeMovingParzenValues[m], fv_mv,
            *imageJacobian, *maskJacobian);
          ++it;
        }
        it.NextLine();
      }
    }

  } // end UpdateJointPDFAndDerivatives


  /**
	 * *************** UpdateJointPDFDerivatives ***************************
	 */

	template < class TFixedImage, class TMovingImage >
		void
		ParzenWindowHistogramImageToImageMetric<TFixedImage,TMovingImage>
		::UpdateJointPDFDerivatives(
    const JointPDFIndexType & pdfIndex, double factor_a, double factor_b,
    const DerivativeType & imageJacobian, const DerivativeType & maskJacobian) const
	{
		/** Get the pointer to the element with index [0, pdfIndex[0], pdfIndex[1]]*/
		JointPDFValueType * derivPtr = this->m_JointPDFDerivatives->GetBufferPointer() +
      ( pdfIndex[0] * this->m_JointPDFDerivatives->GetOffsetTable()[1] ) +
			( pdfIndex[1] * this->m_JointPDFDerivatives->GetOffsetTable()[2] );
    
		if( this->m_NonZeroJacobianIndices.GetSize() == this->m_NumberOfParameters )
		{
			/** Loop over all jacobians */
      typename DerivativeType::const_iterator imjac = imageJacobian.begin();
      typename DerivativeType::const_iterator maskjac = maskJacobian.begin();
			for ( unsigned int mu = 0; mu < this->m_NumberOfParameters; ++mu )
			{
				*(derivPtr) += static_cast<PDFValueType>(
          (*maskjac) * factor_b - (*imjac) * factor_a );
        ++derivPtr;
        ++imjac;
        ++maskjac;
			}
		} 
		else
		{
			/** Loop only over the non-zero jacobians */
			for ( unsigned int i = 0; i < maskJacobian.GetSize(); ++i)
      {
				const unsigned int mu = this->m_NonZeroJacobianIndices[i];
        JointPDFValueType * ptr = derivPtr + mu;
				*(ptr) += static_cast<PDFValueType>(
          maskJacobian[i] * factor_b - imageJacobian[i] * factor_a );
			}
		}
		
	} // end UpdateJointPDFDerivatives
 
 
	/**
	 * *************** EvaluateTransformJacobianInnerProducts ****************
	 */

	template < class TFixedImage, class TMovingImage >
		void
		ParzenWindowHistogramImageToImageMetric<TFixedImage,TMovingImage>
		::EvaluateTransformJacobianInnerProducts( 
		const TransformJacobianType & jacobian, 
		const MovingImageDerivativeType & movingImageDerivative,
    const MovingImageMaskDerivativeType & movingMaskDerivative,
    DerivativeType & imageJacobian,
    DerivativeType & maskJacobian) const
	{
    typedef typename TransformJacobianType::const_iterator JacobianIteratorType;
    typedef typename DerivativeType::iterator              DerivativeIteratorType;
    JacobianIteratorType jac = jacobian.begin();
    imageJacobian.Fill(0.0);
    maskJacobian.Fill(0.0);
    const unsigned int sizeImageJacobian = imageJacobian.GetSize();
    for ( unsigned int dim = 0; dim < FixedImageDimension; dim++ )
    {
      const double imDeriv = movingImageDerivative[dim];
      const double maskDeriv = movingMaskDerivative[dim];
      DerivativeIteratorType imjac = imageJacobian.begin();
      DerivativeIteratorType maskjac = maskJacobian.begin();
      
      for ( unsigned int mu = 0; mu < sizeImageJacobian ; mu++ )
      {
        (*imjac) += (*jac) * imDeriv;
        (*maskjac) += (*jac) * maskDeriv;
        ++imjac;
        ++maskjac;
        ++jac;
      }
    }
	} // end EvaluateTransformJacobianInnerProducts


  /**
   * *********************** NormalizeJointPDF ***********************
   * Multiply the pdf entries by the given normalization factor 
   */

  template < class TFixedImage, class TMovingImage >
		void
		ParzenWindowHistogramImageToImageMetric<TFixedImage,TMovingImage>
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
		ParzenWindowHistogramImageToImageMetric<TFixedImage,TMovingImage>
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
		ParzenWindowHistogramImageToImageMetric<TFixedImage,TMovingImage>
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
	 * *************** UpdateAlphaDerivatives ***************************
	 */

	template < class TFixedImage, class TMovingImage >
		void
		ParzenWindowHistogramImageToImageMetric<TFixedImage,TMovingImage>
		::UpdateAlphaDerivatives( const DerivativeType & maskJacobian ) const
  {
    if( this->m_NonZeroJacobianIndices.GetSize() == this->m_NumberOfParameters )
		{
      /** Loop over all jacobians */
      typename DerivativeType::const_iterator maskjacit = maskJacobian.begin();
      typename DerivativeType::iterator alphaderivit = this->m_AlphaDerivatives.begin();
      for ( unsigned int mu = 0; mu < this->m_NumberOfParameters; ++mu )
      {
        (*alphaderivit) += (*maskjacit);
        ++alphaderivit;
        ++maskjacit;
      }
    }
    else
    {
      /** Only pick the nonzero jacobians */
      for ( unsigned int i = 0; i < maskJacobian.GetSize(); ++i)
      {
        this->m_AlphaDerivatives[ this->m_NonZeroJacobianIndices[i] ] += maskJacobian[i];
      }
    }
  } // end UpdateAlphaDerivatives


  /**
	 * ************************ ComputePDFs **************************
	 */

	template < class TFixedImage, class TMovingImage  >
	  void
	  ParzenWindowHistogramImageToImageMetric<TFixedImage,TMovingImage>
	  ::ComputePDFs( const ParametersType& parameters ) const
	{
    /** Initialize some variables */    
    this->m_JointPDF->FillBuffer( 0.0 );
    this->m_NumberOfPixelsCounted = 0;
    this->m_Alpha = 0.0;
    double sumOfMovingMaskValues = 0.0;
        
    /** Set up the parameters in the transform */
    this->SetTransformParameters( parameters );

    /** Update the imageSampler and get a handle to the sample container. */
    this->GetImageSampler()->Update();
    ImageSampleContainerPointer sampleContainer = this->GetImageSampler()->GetOutput();

    /** Create iterator over the sample container. */
    typename ImageSampleContainerType::ConstIterator fiter;
    typename ImageSampleContainerType::ConstIterator fbegin = sampleContainer->Begin();
    typename ImageSampleContainerType::ConstIterator fend = sampleContainer->End();
            
    /** Loop over sample container and compute contribution of each sample to pdfs */
    for ( fiter = fbegin; fiter != fend; ++fiter )
    {       
      /** Read fixed coordinates and initialize some variables */
      const FixedImagePointType & fixedPoint = (*fiter).Value().m_ImageCoordinates;
      RealType movingImageValue; 
      MovingImagePointType mappedPoint;
                  
      /** Transform point and check if it is inside the bspline support region */
      bool sampleOk = this->TransformPoint( fixedPoint, mappedPoint);

      /** Check if point is inside mask */
      RealType movingMaskValue = 0.0;
      if ( sampleOk ) 
      {
        this->EvaluateMovingMaskValueAndDerivative( mappedPoint, movingMaskValue, 0 );
        const double smallNumber1 = 1e-10;
        sampleOk = movingMaskValue > smallNumber1;
      }

      /** Compute the moving image value and check if the point is
      * inside the moving image buffer */
      if ( sampleOk )
      {
        sampleOk = this->EvaluateMovingImageValueAndDerivative(
          mappedPoint, movingImageValue, 0 );
      }
      
      if( sampleOk )
      {
        this->m_NumberOfPixelsCounted++; 
        sumOfMovingMaskValues += movingMaskValue;

        /** Get the fixed image value */
        RealType fixedImageValue = static_cast<RealType>( (*fiter).Value().m_ImageValue );

        /** Make sure the values fall within the histogram range */
        fixedImageValue = this->GetFixedImageLimiter()->Evaluate(fixedImageValue);
        movingImageValue = this->GetMovingImageLimiter()->Evaluate(movingImageValue);
        
        /** Compute this sample's contribution to the joint distributions. */
        this->UpdateJointPDFAndDerivatives( 
          fixedImageValue, movingImageValue, movingMaskValue, 0, 0);
      }       

    } // end iterating over fixed image spatial sample container for loop
    
    /** Check if enough samples were valid */
    this->CheckNumberOfSamples(
      sampleContainer->Size(), this->m_NumberOfPixelsCounted, sumOfMovingMaskValues);

    /** Compute alpha */
    this->m_Alpha = 1.0 / sumOfMovingMaskValues;
    
  } // end ComputePDFs

  
  /**
	 * ************************ ComputePDFsAndPDFDerivatives *******************
	 */

	template < class TFixedImage, class TMovingImage  >
	  void
	  ParzenWindowHistogramImageToImageMetric<TFixedImage,TMovingImage>
	  ::ComputePDFsAndPDFDerivatives( const ParametersType& parameters ) const
	{
    /** Initialize some variables */
    this->m_JointPDF->FillBuffer( 0.0 );
    this->m_JointPDFDerivatives->FillBuffer( 0.0 );
    this->m_Alpha = 0.0;
    this->m_AlphaDerivatives.Fill(0.0);
    
    this->m_NumberOfPixelsCounted = 0;
    double sumOfMovingMaskValues = 0.0;
        
    /** Arrays that store dM(x)/dmu and dMask(x)/dmu */
    DerivativeType imageJacobian( this->m_NonZeroJacobianIndices.GetSize() );
    DerivativeType maskJacobian( this->m_NonZeroJacobianIndices.GetSize() );
       
    /** Set up the parameters in the transform */
    this->SetTransformParameters( parameters );

    /** Update the imageSampler and get a handle to the sample container. */
    this->GetImageSampler()->Update();
    ImageSampleContainerPointer sampleContainer = this->GetImageSampler()->GetOutput();

    /** Create iterator over the sample container. */
    typename ImageSampleContainerType::ConstIterator fiter;
    typename ImageSampleContainerType::ConstIterator fbegin = sampleContainer->Begin();
    typename ImageSampleContainerType::ConstIterator fend = sampleContainer->End();

    /** Loop over sample container and compute contribution of each sample to pdfs */
    for ( fiter = fbegin; fiter != fend; ++fiter )
    {
      /** Read fixed coordinates and initialize some variables */
      const FixedImagePointType & fixedPoint = (*fiter).Value().m_ImageCoordinates;
      RealType movingImageValue; 
      MovingImagePointType mappedPoint;
      MovingImageDerivativeType movingImageDerivative;
            
      /** Transform point and check if it is inside the bspline support region */
      bool sampleOk = this->TransformPoint( fixedPoint, mappedPoint);
      
      /** Check if point is inside mask */
      RealType movingMaskValue = 0.0;
      MovingImageMaskDerivativeType movingMaskDerivative; 
      if ( sampleOk ) 
      {
        this->EvaluateMovingMaskValueAndDerivative(
          mappedPoint, movingMaskValue, &movingMaskDerivative );
        const double movingMaskDerivativeMagnitude = movingMaskDerivative.GetNorm();
        const double smallNumber1 = 1e-10;
        sampleOk = ( movingMaskValue > smallNumber1 ) ||
          ( movingMaskDerivativeMagnitude > smallNumber1 );
      }
    
      /** Compute the moving image value M(T(x)) and derivative dM/dx and check if
       * the point is inside the moving image buffer */
      if ( sampleOk )
      {
        sampleOk = this->EvaluateMovingImageValueAndDerivative( 
          mappedPoint, movingImageValue, &movingImageDerivative );
      }
            
      if( sampleOk )
      {
        this->m_NumberOfPixelsCounted++; 
        sumOfMovingMaskValues += movingMaskValue;

        /** Get the fixed image value */
        RealType fixedImageValue = static_cast<RealType>( (*fiter).Value().m_ImageValue );

        /** Make sure the values fall within the histogram range */
        fixedImageValue = this->GetFixedImageLimiter()->Evaluate(fixedImageValue);
        movingImageValue = this->GetMovingImageLimiter()->Evaluate(
          movingImageValue, movingImageDerivative );
        
        /** Get the TransformJacobian dT/dmu*/
        const TransformJacobianType & jacobian = 
          this->EvaluateTransformJacobian( fixedPoint );
        
        /** compute the innerproducts (dM/dx)^T (dT/dmu) and (dMask/dx)^T (dT/dmu) */
        this->EvaluateTransformJacobianInnerProducts( 
          jacobian, movingImageDerivative, movingMaskDerivative, imageJacobian, maskJacobian );

        /** Add the maskjacobian to dAlpha/dmu */
        this->UpdateAlphaDerivatives(maskJacobian);
        
        /** Update the joint pdf and the joint pdf derivatives */
        this->UpdateJointPDFAndDerivatives(
          fixedImageValue, movingImageValue, movingMaskValue, &imageJacobian, &maskJacobian );
                              
      } //end if-block check sampleOk
    } // end iterating over fixed image spatial sample container for loop
    
    /** Check if enough samples were valid */
    this->CheckNumberOfSamples(
      sampleContainer->Size(), this->m_NumberOfPixelsCounted, sumOfMovingMaskValues );

    /** Compute alpha and its derivatives */
    this->m_Alpha = 1.0 / sumOfMovingMaskValues;
    this->m_AlphaDerivatives *= - this->m_Alpha * this->m_Alpha;
    
  } // end ComputePDFsAndPDFDerivatives

} // end namespace itk 


#endif // end #ifndef _itkParzenWindowHistogramImageToImageMetric_HXX__

