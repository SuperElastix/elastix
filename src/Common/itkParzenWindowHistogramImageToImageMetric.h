#ifndef __itkParzenWindowHistogramImageToImageMetric_H__
#define __itkParzenWindowHistogramImageToImageMetric_H__

#include "itkAdvancedImageToImageMetric.h"
#include "itkBSplineKernelFunction.h"
#include "itkBSplineDerivativeKernelFunction.h"

namespace itk
{
	
	/**
	 * \class ParzenWindowHistogramImageToImageMetric
	 * \brief A base class for image metrics based on a joint histogram 
   * computed using Parzen Windowing
	 *
	 * The calculations are based on the method of Mattes/Thevenaz/Unser [1,2,3]
	 * where the probability density distribution are estimated using
	 * Parzen histograms. 
   *
	 * One the PDF's have been contructed, the metric value and derivative
   * can be computed. Inheriting classes should make sure to call
   * the function ComputePDFs(AndPDFDerivatives) before using m_JointPDF and m_Alpha
   * (and m_JointPDFDerivatives and m_AlphaDerivatives).
   *
   * This class does not define the GetValue/GetValueAndDerivative methods.
   * This is the task of inheriting classes.
   * 
   * The code is based on the itk::MattesMutualInformationImageToImageMetric,
   * but largely rewritten. Some important features:
   *  - It inherits from AdvancedImageToImageMetric, which provides a lot of 
   *    general functionality.
   *  - It splits up some functions in subfunctions.
   *  - The parzen window order can be chosen.
   *  - A fixed and moving number of histogram bins can be chosen.
   *  - More use of iterators instead of raw buffer pointers.
   *  - An optional FiniteDifference derivative estimation. 
   *
   * \warning This class in not thread safe due the member data structures
	 *  used to the store the sampled points and the marginal and joint pdfs.
	 *
	 * References:\n
	 * [1] "Nonrigid multimodality image registration"\n
	 *      D. Mattes, D. R. Haynor, H. Vesselle, T. Lewellen and W. Eubank\n
	 *      Medical Imaging 2001: Image Processing, 2001, pp. 1609-1620.\n
	 * [2] "PET-CT Image Registration in the Chest Using Free-form Deformations"\n
	 *      D. Mattes, D. R. Haynor, H. Vesselle, T. Lewellen and W. Eubank\n
	 *      IEEE Transactions in Medical Imaging. To Appear.\n
	 * [3] "Optimization of Mutual Information for MultiResolution Image
	 *      Registration"\n
	 *      P. Thevenaz and M. Unser\n
	 *      IEEE Transactions in Image Processing, 9(12) December 2000.\n
	 *
	 *
	 * \ingroup Metrics
	 */

	template <class TFixedImage,class TMovingImage >
		class ParzenWindowHistogramImageToImageMetric :
	public AdvancedImageToImageMetric< TFixedImage, TMovingImage >
	{
	public:
		
		/** Standard class typedefs. */
		typedef ParzenWindowHistogramImageToImageMetric					        Self;
		typedef AdvancedImageToImageMetric< TFixedImage, TMovingImage >			Superclass;
		typedef SmartPointer<Self>																	        Pointer;
		typedef SmartPointer<const Self>														        ConstPointer;
				
		/** Run-time type information (and related methods). */
		itkTypeMacro( ParzenWindowHistogramImageToImageMetric, AdvancedImageToImageMetric );

    /** Typedefs from the superclass. */
    typedef typename 
      Superclass::CoordinateRepresentationType              CoordinateRepresentationType;
    typedef typename Superclass::MovingImageType            MovingImageType;
    typedef typename Superclass::MovingImagePixelType       MovingImagePixelType;
    typedef typename Superclass::MovingImageConstPointer    MovingImageConstPointer;
    typedef typename Superclass::FixedImageType             FixedImageType;
    typedef typename Superclass::FixedImageConstPointer     FixedImageConstPointer;
    typedef typename Superclass::FixedImageRegionType       FixedImageRegionType;
    typedef typename Superclass::TransformType              TransformType;
    typedef typename Superclass::TransformPointer           TransformPointer;
    typedef typename Superclass::InputPointType             InputPointType;
    typedef typename Superclass::OutputPointType            OutputPointType;
    typedef typename Superclass::TransformParametersType    TransformParametersType;
    typedef typename Superclass::TransformJacobianType      TransformJacobianType;
    typedef typename Superclass::InterpolatorType           InterpolatorType;
    typedef typename Superclass::InterpolatorPointer        InterpolatorPointer;
    typedef typename Superclass::RealType                   RealType;
    typedef typename Superclass::GradientPixelType          GradientPixelType;
    typedef typename Superclass::GradientImageType          GradientImageType;
    typedef typename Superclass::GradientImagePointer       GradientImagePointer;
    typedef typename Superclass::GradientImageFilterType    GradientImageFilterType;
    typedef typename Superclass::GradientImageFilterPointer GradientImageFilterPointer;
    typedef typename Superclass::FixedImageMaskType         FixedImageMaskType;
    typedef typename Superclass::FixedImageMaskPointer      FixedImageMaskPointer;
    typedef typename Superclass::MovingImageMaskType        MovingImageMaskType;
    typedef typename Superclass::MovingImageMaskPointer     MovingImageMaskPointer;
    typedef typename Superclass::MeasureType                MeasureType;
    typedef typename Superclass::DerivativeType             DerivativeType;
    typedef typename Superclass::ParametersType             ParametersType;
    typedef typename Superclass::FixedImagePixelType        FixedImagePixelType;
    typedef typename Superclass::MovingImageRegionType      MovingImageRegionType;
    typedef typename Superclass::ImageSamplerType           ImageSamplerType;
    typedef typename Superclass::ImageSamplerPointer        ImageSamplerPointer;
    typedef typename Superclass::ImageSampleContainerType   ImageSampleContainerType;
    typedef typename 
      Superclass::ImageSampleContainerPointer               ImageSampleContainerPointer;
    typedef typename Superclass::InternalMaskPixelType      InternalMaskPixelType;
    typedef typename
      Superclass::InternalMovingImageMaskType               InternalMovingImageMaskType;
    typedef typename 
      Superclass::MovingImageMaskInterpolatorType           MovingImageMaskInterpolatorType;
    typedef typename Superclass::FixedImageLimiterType      FixedImageLimiterType;
    typedef typename Superclass::MovingImageLimiterType     MovingImageLimiterType;
    typedef typename
      Superclass::FixedImageLimiterOutputType               FixedImageLimiterOutputType;
    typedef typename
      Superclass::MovingImageLimiterOutputType              MovingImageLimiterOutputType;
		
    /** The fixed image dimension. */
		itkStaticConstMacro( FixedImageDimension, unsigned int,
			FixedImageType::ImageDimension );

		/** The moving image dimension. */
		itkStaticConstMacro( MovingImageDimension, unsigned int,
			MovingImageType::ImageDimension );
 
		/** Initialize the Metric by
		 * (1) Call the superclass' implementation
		 * (2) InitializeHistograms()
     * (3) InitializeKernels() 
     * (4) Resize AlphaDerivatives */
		void Initialize(void) throw ( ExceptionObject );
				
		/** Get the derivatives of the match measure. This method simply calls the 
     * the GetValueAndDerivative, since this will be mostly almost as fast 
     * as just computing the derivative. */
		void GetDerivative( 
			const ParametersType& parameters,
			DerivativeType & Derivative ) const;

    /**  Get the value and derivatives for single valued optimizers.
     * This method calls this->GetValueAndAnalyticDerivative or
     * this->GetValueAndFiniteDifferenceDerivative, depending on the bool
     * m_UseFiniteDifferenceDerivative */
		void GetValueAndDerivative( const ParametersType& parameters, 
			MeasureType& value, DerivativeType& derivative ) const; 
				
		/** Number of bins to use for the fixed image in the histogram. Typical value is 50. */
		itkSetClampMacro( NumberOfFixedHistogramBins, unsigned long,
			1, NumericTraits<unsigned long>::max() );
		itkGetMacro( NumberOfFixedHistogramBins, unsigned long);   

    /** Number of bins for the moving image to use in the histogram. Typical value is 50. */
		itkSetClampMacro( NumberOfMovingHistogramBins, unsigned long,
			1, NumericTraits<unsigned long>::max() );
		itkGetMacro( NumberOfMovingHistogramBins, unsigned long);   

    /** The bspline order of the fixed parzen window; default: 0 */
    itkSetClampMacro(FixedKernelBSplineOrder, unsigned int, 0, 3);
    itkGetConstMacro(FixedKernelBSplineOrder, unsigned int);
    
    /** The bspline order of the moving bspline order; default: 3 */
    itkSetClampMacro(MovingKernelBSplineOrder, unsigned int, 0, 3);
    itkGetConstMacro(MovingKernelBSplineOrder, unsigned int);

    /** Whether you plan to call the GetDerivative/GetValueAndDerivative method or not
     * This option should be set before calling Initialize()
     * Default: false */
    itkSetMacro(UseDerivative, bool);
    itkGetConstMacro(UseDerivative, bool);

    /** Whether you want to use a finite difference implementation of the metric's derivative.
     * This option should be set before calling Initialize(); default: false */
    itkSetMacro(UseFiniteDifferenceDerivative, bool);
    itkGetConstMacro(UseFiniteDifferenceDerivative, bool);

    /** For computing the finite difference derivative, the perturbation (delta) of the 
     * transform parameters; default: 1.0.
     * mu_right= mu + delta*e_k */
    itkSetMacro( FiniteDifferencePerturbation, double );
    itkGetConstMacro( FiniteDifferencePerturbation, double );
    
	protected:
		
		/** The constructor. */
		ParzenWindowHistogramImageToImageMetric();

		/** The destructor. */
		virtual ~ParzenWindowHistogramImageToImageMetric() {};

		/** Print Self. */
		void PrintSelf( std::ostream& os, Indent indent ) const;

    /** Protected Typedefs ******************/
  
    /** Typedefs inherited from superclass */
    typedef typename Superclass::FixedImageIndexType                FixedImageIndexType;
	  typedef typename Superclass::FixedImageIndexValueType           FixedImageIndexValueType;
	  typedef typename Superclass::MovingImageIndexType               MovingImageIndexType;
	  typedef typename Superclass::FixedImagePointType                FixedImagePointType;
	  typedef typename Superclass::MovingImagePointType               MovingImagePointType;
    typedef typename Superclass::MovingImageContinuousIndexType     MovingImageContinuousIndexType;
  	typedef	typename Superclass::BSplineInterpolatorType            BSplineInterpolatorType;
    typedef typename Superclass::MovingImageDerivativeType          MovingImageDerivativeType;
    typedef typename Superclass::CentralDifferenceGradientFilterType        CentralDifferenceGradientFilterType;
    typedef typename Superclass::BSplineTransformType               BSplineTransformType;
    typedef typename Superclass::BSplineTransformWeightsType        BSplineTransformWeightsType;
	  typedef typename Superclass::BSplineTransformIndexArrayType     BSplineTransformIndexArrayType;
	  typedef typename Superclass::BSplineCombinationTransformType    BSplineCombinationTransformType;
 	  typedef typename Superclass::BSplineParametersOffsetType        BSplineParametersOffsetType;
    typedef typename Superclass::ParameterIndexArrayType            ParameterIndexArrayType;
    typedef typename Superclass::MovingImageMaskDerivativeType      MovingImageMaskDerivativeType;
        
    /** Typedefs for the PDFs and PDF derivatives.  */
    typedef float                                   PDFValueType;
		typedef Array<PDFValueType>                     MarginalPDFType;
    typedef Image<PDFValueType,2>									  JointPDFType;
    typedef Image<PDFValueType,3>									  JointPDFDerivativesType;
    typedef Image<PDFValueType,2>                   IncrementalMarginalPDFType;
		typedef JointPDFType::IndexType                 JointPDFIndexType;
    typedef JointPDFType::RegionType                JointPDFRegionType;
		typedef JointPDFType::SizeType                  JointPDFSizeType;
		typedef JointPDFDerivativesType::IndexType      JointPDFDerivativesIndexType;
    typedef JointPDFDerivativesType::RegionType     JointPDFDerivativesRegionType;
		typedef JointPDFDerivativesType::SizeType       JointPDFDerivativesSizeType;
    typedef IncrementalMarginalPDFType::IndexType   IncrementalMarginalPDFIndexType;
    typedef IncrementalMarginalPDFType::RegionType  IncrementalMarginalPDFRegionType;
		typedef IncrementalMarginalPDFType::SizeType    IncrementalMarginalPDFSizeType;
		typedef Array<double>                           ParzenValueContainerType;
    
    /** Typedefs for parzen kernel . */
    typedef KernelFunction KernelFunctionType;

    /** Protected variables **************************** */
   		
    /** Variables for Alpha and dAlpha/dmu (the normalization factor of the histogram) */
    mutable double                                m_Alpha;
    mutable DerivativeType                        m_AlphaDerivatives;
    mutable DerivativeType                        m_PerturbedAlphaRight;
    mutable DerivativeType                        m_PerturbedAlphaLeft;
   
    /** Variables for the pdfs (actually: histograms) */
    mutable MarginalPDFType                       m_FixedImageMarginalPDF;
		mutable MarginalPDFType                       m_MovingImageMarginalPDF;
    typename JointPDFType::Pointer								m_JointPDF;
		typename JointPDFDerivativesType::Pointer			m_JointPDFDerivatives;
    typename JointPDFDerivativesType::Pointer     m_IncrementalJointPDFRight;
    typename JointPDFDerivativesType::Pointer     m_IncrementalJointPDFLeft;
    typename IncrementalMarginalPDFType::Pointer  m_FixedIncrementalMarginalPDFRight;
    typename IncrementalMarginalPDFType::Pointer  m_MovingIncrementalMarginalPDFRight;
    typename IncrementalMarginalPDFType::Pointer  m_FixedIncrementalMarginalPDFLeft;
    typename IncrementalMarginalPDFType::Pointer  m_MovingIncrementalMarginalPDFLeft;
    mutable JointPDFRegionType                    m_JointPDFWindow;
    double m_MovingImageNormalizedMin;
		double m_FixedImageNormalizedMin;
		double m_FixedImageBinSize;
		double m_MovingImageBinSize;
    double m_FixedParzenTermToIndexOffset;
    double m_MovingParzenTermToIndexOffset;
    
    /** Kernels for computing Parzen histograms and derivatives. */
		typename KernelFunctionType::Pointer m_FixedKernel;
    typename KernelFunctionType::Pointer m_MovingKernel;
		typename KernelFunctionType::Pointer m_DerivativeMovingKernel;

    /** Computes the innerproduct of transform jacobian with moving image gradient
     * and transform jacobian with the derivative of the movingMask
     * The results are stored in imageJacobian and maskJacobian, which are supposed
     * to have the right size (same length as jacobian's number of columns). */
    void EvaluateTransformJacobianInnerProducts(
      const TransformJacobianType & jacobian, 
		  const MovingImageDerivativeType & movingImageDerivative,
      const MovingImageMaskDerivativeType & movingMaskDerivative,
      DerivativeType & imageJacobian,
      DerivativeType & maskJacobian) const;
 
    /** Compute the parzen values given an image value and a starting histogram index
     * Compute the values at (parzenWindowIndex - parzenWindowTerm + k) for 
     * k = 0 ... kernelsize-1
     * Returns the values in a ParzenValueContainer, which is supposed to have
     * the right size already **/
    void EvaluateParzenValues(
      double parzenWindowTerm, int parzenWindowIndex,
      const KernelFunctionType * kernel, ParzenValueContainerType & parzenValues) const;
    
    /** Update the joint PDF with a pixel pair; on demand also updates the 
     * pdf derivatives (if the jacobian pointers are nonzero) */
    virtual void UpdateJointPDFAndDerivatives(
      RealType fixedImageValue, RealType movingImageValue, RealType movingMaskValue,
      const DerivativeType * imageJacobian, const DerivativeType * maskJacobian) const;

    /** Update the joint PDF and the incremental pdfs.
     * The input is a pixel pair (fixed, moving, moving mask) and
     * a set of moving image/mask values when using mu+delta*e_k, for 
     * each k that has a nonzero jacobian. And for mu-delta*e_k of course.
     * Also updates the PerturbedAlpha's/
     */
    virtual void UpdateJointPDFAndIncrementalPDFs(
      RealType fixedImageValue, RealType movingImageValue, RealType movingMaskValue,
      const DerivativeType & movingImageValuesRight,
      const DerivativeType & movingImageValuesLeft,
      const DerivativeType & movingMaskValuesRight,
      const DerivativeType & movingMaskValuesLeft ) const;

    /** Update the pdf derivatives
     * adds -image_jac[mu]*factor_a + mask_jac[mu]*factor_b to the bin 
     * with index [ mu, pdfIndex[0], pdfIndex[1] ] for all mu.
     * This function should only be called from UpdateJointPDFAndDerivatives */
    void UpdateJointPDFDerivatives(
      const JointPDFIndexType & pdfIndex, double factor_a, double factor_b,
      const DerivativeType & imageJacobian, const DerivativeType & maskJacobian) const;

    /** Adds the MaskJacobian to the alpha derivative vector */
    virtual void UpdateAlphaDerivatives(const DerivativeType & maskJacobian) const;	

    /** Multiply the pdf entries by the given normalization factor */
    virtual void NormalizeJointPDF(
      JointPDFType * pdf, double factor ) const;

    /** Multiply the pdf derivatives entries by the given normalization factor */
    virtual void NormalizeJointPDFDerivatives(
      JointPDFDerivativesType * pdf, double factor ) const;

    /** Compute marginal pdfs by summing over the joint pdf
     * direction = 0: fixed marginal pdf
     * direction = 1: moving marginal pdf */
    virtual void ComputeMarginalPDF( 
      const JointPDFType * jointPDF,
      MarginalPDFType & marginalPDF,
      unsigned int direction ) const;

    /** Compute incremental marginal pdfs. Integrates the incremental PDF
     * to obtain the fixed and moving marginal pdfs at once */
    virtual void ComputeIncrementalMarginalPDFs( 
      const JointPDFDerivativesType * incrementalPDF, 
      IncrementalMarginalPDFType * fixedIncrementalMarginalPDF,
      IncrementalMarginalPDFType * movingIncrementalMarginalPDF ) const;   

    /** Compute PDFs and pdf derivatives; Loops over the fixed image samples and constructs
     * the m_JointPDF, m_JointPDFDerivatives, m_Alpha and m_AlphaDerivatives
     * The JointPDF and Alpha and its derivatives are related as follows:
     * p = m_Alpha * m_JointPDF
     * dp/dmu = m_AlphaDerivatives * m_JointPDF + m_Alpha * m_JointPDFDerivatives
     * So, the JointPDF is more like a histogram than a true pdf...
     * The histograms are left unnormalised since it may be faster to 
     * not do this explicitly. */
    virtual void ComputePDFsAndPDFDerivatives( const ParametersType & parameters) const;

    /** Compute PDFs and incremental pdfs (which you can use to compute finite
     * difference estimate of the derivative).
     * Loops over the fixed image samples and constructs the m_JointPDF,
     * m_IncrementalJointPDF<Right/Left>, m_Alpha, and m_PerturbedAlpha<Right/Left>.
     *
     * mu = input parameters vector
     * jh(mu) = m_JointPDF(:,:) = joint histogram
     * ihr(k) = m_IncrementalJointPDFRight(k,:,:)
     * ihl(k) = m_IncrementalJointPDFLeft(k,:,:)
     * a(mu) = m_Alpha
     * par(k) = m_PerturbedAlphaRight(k)
     * pal(k) = m_PerturbedAlphaLeft(k)
     * size(ihr) = = size(ihl) = nrofparams * nrofmovingbins * nroffixedbins
     * 
     * ihr and ihl are determined such that:
     * jh(mu+delta*e_k) = jh(mu) + ihr(k)
     * jh(mu-delta*e_k) = jh(mu) + ihl(k)
     * where e_k is the unit vector.
     * 
     * the pdf can be derived with:
     * p(mu+delta*e_k) = ( par(k) ) * jh(mu+delta*e_k)
     * p(mu-delta*e_k) = ( pal(k) ) * jh(mu-delta*e_k)
     */
    virtual void ComputePDFsAndIncrementalPDFs( const ParametersType & parameters) const;

    /** Compute PDFs; Loops over the fixed image samples and constructs
     * the m_JointPDF and m_Alpha
     * The JointPDF and Alpha are related as follows:
     * p = m_Alpha * m_JointPDF 
     * So, the JointPDF is more like a histogram than a true pdf...
     * The histogram is left unnormalised since it may be faster to 
     * not do this explicitly. */
    virtual void ComputePDFs( const ParametersType & parameters) const;

    /** Some initialization functions, called by Initialize. */
    virtual void InitializeHistograms(void);
    virtual void InitializeKernels(void);

    /**  Get the value and analytic derivatives for single valued optimizers.
     * Called by GetValueAndDerivative if UseFiniteDifferenceDerivative == false
     * Implement this method in subclasses */
    virtual void GetValueAndAnalyticDerivative( const ParametersType& parameters, 
			MeasureType& value, DerivativeType& derivative ) const {};

    /**  Get the value and finite difference derivatives for single valued optimizers.
     * Called by GetValueAndDerivative if UseFiniteDifferenceDerivative == true
     * Implement this method in subclasses */
		virtual void GetValueAndFiniteDifferenceDerivative( const ParametersType& parameters, 
			MeasureType& value, DerivativeType& derivative ) const {};   

	private:
		
		/** The private constructor. */
		ParzenWindowHistogramImageToImageMetric( const Self& );	// purposely not implemented
		/** The private copy constructor. */
		void operator=( const Self& );															// purposely not implemented
    				
    /** Variables that can/should be accessed by their Set/Get functions. */
		unsigned long m_NumberOfFixedHistogramBins;
    unsigned long m_NumberOfMovingHistogramBins;
    unsigned int m_FixedKernelBSplineOrder;
    unsigned int m_MovingKernelBSplineOrder;
    bool m_UseDerivative;
    bool m_UseFiniteDifferenceDerivative;
    double m_FiniteDifferencePerturbation;
		
	}; // end class ParzenWindowHistogramImageToImageMetric

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkParzenWindowHistogramImageToImageMetric.hxx"
#endif

#endif // end #ifndef __itkParzenWindowHistogramImageToImageMetric_H__

