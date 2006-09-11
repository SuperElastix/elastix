#ifndef __itkMattesMutualInformationImageToImageMetric3_H__
#define __itkMattesMutualInformationImageToImageMetric3_H__

#include "itkImageToImageMetricWithSampling.h"
#include "itkCovariantVector.h"
#include "itkPoint.h"
#include "itkIndex.h"
#include "itkBSplineKernelFunction.h"
#include "itkBSplineDerivativeKernelFunction.h"
#include "itkCentralDifferenceImageFunction.h"
#include "itkBSplineInterpolateImageFunction.h"
#include "itkBSplineDeformableTransform.h"
#include "itkBSplineResampleImageFunction.h"
#include "itkBSplineCombinationTransform.h"

namespace itk
{
	
	/**
	 * \class MattesMutualInformationImageToImageMetric3
	 * \brief Computes the mutual information between two images to be 
	 * registered using the methof of Mattes et al.
	 *
	 * MattesMutualInformationImageToImageMetric computes the mutual 
	 * information between a fixed and moving image to be registered.
	 *
	 * This class is templated over the FixedImage type and the MovingImage 
	 * type.
	 *
	 * The fixed and moving images are set via methods SetFixedImage() and
	 * SetMovingImage(). This metric makes use of user specified Transform and
	 * Interpolator. The Transform is used to map points from the fixed image to
	 * the moving image domain. The Interpolator is used to evaluate the image
	 * intensity at user specified geometric points in the moving image.
	 * The Transform and Interpolator are set via methods SetTransform() and
	 * SetInterpolator().
	 *
	 * If a BSplineInterpolationFunction is used, this class obtain
	 * image derivatives from the BSpline interpolator. Otherwise, 
	 * image derivatives are computed using central differencing.
	 *
	 * \warning This metric assumes that the moving image has already been
	 * connected to the interpolator outside of this class. 
	 *
	 * The method GetValue() computes of the mutual information
	 * while method GetValueAndDerivative() computes
	 * both the mutual information and its derivatives with respect to the
	 * transform parameters.
	 *
	 * The calculations are based on the method of Mattes et al [1,2]
	 * where the probability density distribution are estimated using
	 * Parzen histograms. Since the fixed image PDF does not contribute
	 * to the derivatives, it does not need to be smooth. Hence, 
	 * a zero order (box car) BSpline kernel is used
	 * for the fixed image intensity PDF. On the other hand, to ensure
	 * smoothness a third order BSpline kernel is used for the 
	 * moving image intensity PDF.
	 *
	 * On Initialize(), the FixedImage is uniformly sampled within
	 * the FixedImageRegion. The number of samples used can be set
	 * via SetNumberOfSpatialSamples(). Typically, the number of
	 * spatial samples used should increase with the image size.
	 *
	 * During each call of GetValue(), GetDerivatives(),
	 * GetValueAndDerivatives(), marginal and joint intensity PDF's
	 * values are estimated at discrete position or bins. 
	 * The number of bins used can be set via SetNumberOfHistogramBins().
	 * To handle data with arbitray magnitude and dynamic range, 
	 * the image intensity is scale such that any contribution to the
	 * histogram will fall into a valid bin.
	 *
	 * One the PDF's have been contructed, the mutual information
	 * is obtained by doubling summing over the discrete PDF values.
	 *
	 *
	 * Notes: 
	 * 1. This class returns the negative mutual information value.
	 * 2. This class in not thread safe due the private data structures
	 *     used to the store the sampled points and the marginal and joint pdfs.
	 *
	 * References:
	 * [1] "Nonrigid multimodality image registration"
	 *      D. Mattes, D. R. Haynor, H. Vesselle, T. Lewellen and W. Eubank
	 *      Medical Imaging 2001: Image Processing, 2001, pp. 1609-1620.
	 * [2] "PET-CT Image Registration in the Chest Using Free-form Deformations"
	 *      D. Mattes, D. R. Haynor, H. Vesselle, T. Lewellen and W. Eubank
	 *      IEEE Transactions in Medical Imaging. To Appear.
	 * [3] "Optimization of Mutual Information for MultiResolution Image
	 *      Registration"
	 *      P. Thevenaz and M. Unser
	 *      IEEE Transactions in Image Processing, 9(12) December 2000.
	 *
	 *
	 *	NB:
	 * This file declares the itk::MattesMutualInformationImageToImageMetric3.
	 * It is largely the same as itk::MattesMutualInformationImageToImageMetric.
   * For Elastix the following things have been added/changed
   *  - It inherits from ImageToImageMetricWithSampling, which
   *    replaces/enhances the SampleFixedImageDomain functionality.
   *  - It is not only optimised for BSplineTransforms, but also
   *    for the BSplineCombinationTransform.
 	 *
	 * \ingroup Metrics
	 */

	template <class TFixedImage,class TMovingImage >
		class MattesMutualInformationImageToImageMetric3 :
	public ImageToImageMetricWithSampling< TFixedImage, TMovingImage >
	{
	public:
		
		/** Standard class typedefs. */
		typedef MattesMutualInformationImageToImageMetric3					Self;
		typedef ImageToImageMetricWithSampling< TFixedImage, TMovingImage >			Superclass;
		typedef SmartPointer<Self>																	Pointer;
		typedef SmartPointer<const Self>														ConstPointer;
		
		/** Method for creation through the object factory. */
		itkNewMacro( Self );
		
		/** Run-time type information (and related methods). */
		itkTypeMacro( MattesMutualInformationImageToImageMetric3, ImageToImageMetricWithSampling );
		
		/** Types inherited from Superclass. */
		typedef typename Superclass::TransformType            TransformType;
		typedef typename Superclass::TransformPointer         TransformPointer;
		typedef typename Superclass::TransformJacobianType    TransformJacobianType;
		typedef typename Superclass::InterpolatorType         InterpolatorType;
		typedef typename Superclass::MeasureType              MeasureType;
		typedef typename Superclass::DerivativeType           DerivativeType;
		typedef typename Superclass::ParametersType           ParametersType;
		typedef typename Superclass::FixedImageType           FixedImageType;
		typedef typename Superclass::MovingImageType          MovingImageType;
		typedef typename Superclass::FixedImageConstPointer   FixedImageConstPointer;
		typedef typename Superclass::MovingImageConstPointer  MovingImageConstPointer;
		typedef typename Superclass::CoordinateRepresentationType
			CoordinateRepresentationType;

    /** Typedefs for indices and points */
		typedef typename FixedImageType::IndexType            FixedImageIndexType;
		typedef typename FixedImageIndexType::IndexValueType  FixedImageIndexValueType;
		typedef typename MovingImageType::IndexType           MovingImageIndexType;
		typedef typename TransformType::InputPointType        FixedImagePointType;
		typedef typename TransformType::OutputPointType       MovingImagePointType;

    /** Typedefs for ImageSampler support */
    typedef typename Superclass::ImageSamplerType              ImageSamplerType;
    typedef typename Superclass::ImageSamplerPointer           ImageSamplerPointer;
    typedef typename Superclass::ImageSampleContainerType      ImageSampleContainerType;
    typedef typename Superclass::ImageSampleContainerPointer   ImageSampleContainerPointer;
		
    /** The fixed image dimension. */
		itkStaticConstMacro( FixedImageDimension, unsigned int,
			FixedImageType::ImageDimension );

		/** The moving image dimension. */
		itkStaticConstMacro( MovingImageDimension, unsigned int,
			MovingImageType::ImageDimension );

    /** Typedefs for mask support */
    typedef unsigned char                                   InternalMaskPixelType;
    typedef typename itk::Image<
      InternalMaskPixelType, 
      itkGetStaticConstMacro(MovingImageDimension) >        InternalMovingImageMaskType;
    typedef typename MovingImageType::SpacingType           MovingImageSpacingType;
    typedef itk::BSplineResampleImageFunction<
      InternalMovingImageMaskType,
      CoordinateRepresentationType >                        MovingImageMaskInterpolatorType;
    typedef typename 
      MovingImageMaskInterpolatorType::CovariantVectorType  MovingImageMaskDerivativeType;
    typedef typename 
      MovingImageMaskInterpolatorType::ContinuousIndexType  MovingImageContinuousIndexType;
		
		/** Initialize the Metric by
		 * (1) making sure that all the components are present and plugged
		 *     together correctly,
		 * (2) allocate memory for pdf data structures.
		 */
		void Initialize(void) throw ( ExceptionObject );
				
		/** Get the derivatives of the match measure. */
		void GetDerivative( 
			const ParametersType& parameters,
			DerivativeType & Derivative ) const;
		
		/**  Get the value. */
		MeasureType GetValue( const ParametersType& parameters ) const;

		/**  Get the value and derivatives for single valued optimizers. */
		void GetValueAndDerivative( const ParametersType& parameters, 
			MeasureType& Value, DerivativeType& Derivative ) const;

    /** Get the internal moving image mask. Equals the movingimage mask if set, and 
     * otherwise it's a box with size equal to the moving image's largest possible region */
    itkGetConstObjectMacro(InternalMovingImageMask, InternalMovingImageMaskType);

    /** Get the interpolator of the internal moving image mask */
    itkGetConstObjectMacro(MovingImageMaskInterpolator, MovingImageMaskInterpolatorType);

    /** Set/Get whether the overlap should be taken into account while computing the derivative
     * This setting also affects the value of the metric. Default: true; */
    itkSetMacro(UseDifferentiableOverlap, bool);
    itkGetConstMacro(UseDifferentiableOverlap, bool);
		
		/** Number of bins to use for the fixed image in the histogram. Typical value is 50. */
		itkSetClampMacro( NumberOfFixedHistogramBins, unsigned long,
			1, NumericTraits<unsigned long>::max() );
		itkGetMacro( NumberOfFixedHistogramBins, unsigned long);   

    /** Number of bins for the moving image to use in the histogram. Typical value is 50. */
		itkSetClampMacro( NumberOfMovingHistogramBins, unsigned long,
			1, NumericTraits<unsigned long>::max() );
		itkGetMacro( NumberOfMovingHistogramBins, unsigned long);   

    /** Setting whether to check if enough samples map inside the moving image. Default: true */
    itkSetMacro(CheckNumberOfSamples, bool);  
    itkGetConstMacro(CheckNumberOfSamples, bool);  

    /** Set the interpolation spline order for the moving image mask; default: 2
     * Make sure to call this before calling Initialize(), if you want to change it. */
    virtual void SetMovingImageMaskInterpolationOrder(unsigned int order)
    {
      this->m_MovingImageMaskInterpolator->SetSplineOrder( order );
    };
    /** Get the interpolation spline order for the moving image mask */
    virtual const unsigned int GetMovingImageMaskInterpolationOrder(void) const
    {
      return this->m_MovingImageMaskInterpolator->GetSplineOrder();
    };

    /** Use a hard limiter for the moving gray values; default: false;
     * Make sure to set either HardLimitMovingGrayValues or SoftLimitGrayValue
     * to true, or both to false; both true yields a soft limiter.
     * If you use a neirest neighbor or linear interpolator, set the LimitRangeRatio
     * to zero and use a hard limiter, or no limiter at all.
     * The hardlimiter does the following to the moving image values:
     * imagevalue = min( imagevalue, maxlimit )
     * imagevalue = max( imagevalue, minlimit )
     * where max/minlimit are defined by the limitrangeratio.
     *
     * NB: For the fixed image always a hard limiter is used.
     */
    itkSetMacro(HardLimitMovingGrayValues, bool);
    itkGetConstMacro(HardLimitMovingGrayValues, bool);

    /** Use a soft limiter for the moving gray values; default: true;
     * Make sure to set either HardLimitMovingGrayValues or SoftLimitGrayValue
     * to true, or both to false; both true yields a soft limiter.
     * If you use a neirest neighbor or linear interpolator, set the LimitRangeRatio
     * to zero and use a hard limiter, or no limiter at all.
     * the softlimiter applies the following intensity transform to the 
     * moving image values, if it's above/under the true image maximum or minimum T:
     * T = true image maximum or minimum
     * L = max or minlimit (as defined by the LimitRangeRatio)
     * a = 1.0 / (T-L)
     * A = 1.0 / ( a e^(aT) )
     * imagevalue = A e ^(a*imagevalue) + L
     * and adapts the image derivatives correspondingly.
     *
     * NB: For the fixed image always a hard limiter is used
     */
    itkSetMacro(SoftLimitMovingGrayValues, bool);
    itkGetConstMacro(SoftLimitMovingGrayValues, bool);

    /** A percentage that defines how much the gray value range is extended
     * maxlimit = max + LimitRangeRatio * (max - min)
     * minlimit = min - LimitRangeRatio * (max - min)
     * Default: 0.01;
     * If you use a neirest neighbor or linear interpolator, set it to zero and
     * use a hard limiter, or no limiter at all.
     * For the fixed image always a hard limiter is used.
     */
    itkSetMacro(MovingLimitRangeRatio, double);
    itkGetConstMacro(MovingLimitRangeRatio, double);
    itkSetMacro(FixedLimitRangeRatio, double);
    itkGetConstMacro(FixedLimitRangeRatio, double);


    /** The bspline order of the fixed parzen window; default: 0 */
    itkSetMacro(FixedKernelBSplineOrder, unsigned int);
    itkGetConstMacro(FixedKernelBSplineOrder, unsigned int);
    
    /** The bspline order of the moving bspline order; default: 3 */
    itkSetMacro(MovingKernelBSplineOrder, unsigned int);
    itkGetConstMacro(MovingKernelBSplineOrder, unsigned int);

	protected:
		
		/** The constructor. */
		MattesMutualInformationImageToImageMetric3();

		/** The destructor. */
		virtual ~MattesMutualInformationImageToImageMetric3() {};

		/** Print Self. */
		void PrintSelf( std::ostream& os, Indent indent ) const;
  
    /** Typedefs used for computing image derivatives */
		typedef	BSplineInterpolateImageFunction<
      MovingImageType, CoordinateRepresentationType>              BSplineInterpolatorType;
  	typedef CentralDifferenceImageFunction<
      MovingImageType, CoordinateRepresentationType>              DerivativeFunctionType;
    typedef typename BSplineInterpolatorType::CovariantVectorType ImageDerivativesType;

    /** Typedefs for BSplineTransform */
    enum { DeformationSplineOrder = 3 };
		typedef BSplineDeformableTransform<
			CoordinateRepresentationType,
			::itk::GetImageDimension<FixedImageType>::ImageDimension,
			DeformationSplineOrder>													            BSplineTransformType;
    typedef typename 
			BSplineTransformType::WeightsType								            BSplineTransformWeightsType;
		typedef typename 
			BSplineTransformType::ParameterIndexArrayType 	            BSplineTransformIndexArrayType;
		typedef itk::BSplineCombinationTransform<
			CoordinateRepresentationType,
			::itk::GetImageDimension<FixedImageType>::ImageDimension,
			DeformationSplineOrder>													            BSplineCombinationTransformType;
 		typedef FixedArray< unsigned long, 
			::itk::GetImageDimension<FixedImageType>::ImageDimension>   ParametersOffsetType;

    /** Typedefs for the PDFs and PDF derivatives.  */
    typedef float                                 PDFValueType;
		typedef Array<PDFValueType>                   MarginalPDFType;
    typedef Image<PDFValueType,2>									JointPDFType;
		typedef Image<PDFValueType,3>									JointPDFDerivativesType;
		typedef JointPDFType::IndexType               JointPDFIndexType;
    typedef JointPDFType::PixelType               JointPDFValueType;
		typedef JointPDFType::RegionType              JointPDFRegionType;
		typedef JointPDFType::SizeType                JointPDFSizeType;
		typedef JointPDFDerivativesType::IndexType    JointPDFDerivativesIndexType;
    typedef JointPDFDerivativesType::PixelType    JointPDFDerivativesValueType;
		typedef JointPDFDerivativesType::RegionType   JointPDFDerivativesRegionType;
		typedef JointPDFDerivativesType::SizeType     JointPDFDerivativesSizeType;
    typedef Array<double>                         ParzenValueContainerType;
    
    /** Typedefs for parzen kernel . */
    typedef KernelFunction KernelFunctionType;

    /** Array type for holding parameter indices */
    typedef Array<unsigned int>                   ParameterIndexArrayType;
    		
    /** Variables for image derivative computation */
		bool m_InterpolatorIsBSpline;
		typename BSplineInterpolatorType::Pointer m_BSplineInterpolator;
		typename DerivativeFunctionType::Pointer  m_DerivativeCalculator;

    /** Variables used when the transform is a bspline transform */
    bool m_TransformIsBSpline;
		bool m_TransformIsBSplineCombination;
    typename BSplineTransformType::Pointer						m_BSplineTransform;
		mutable BSplineTransformWeightsType								m_BSplineTransformWeights;
		mutable BSplineTransformIndexArrayType						m_BSplineTransformIndices;
		typename BSplineCombinationTransformType::Pointer m_BSplineCombinationTransform;
		ParametersOffsetType                              m_ParametersOffset;
		/** The number of BSpline parameters per image dimension. */
		long m_NumParametersPerDim;
		/** The number of BSpline transform weights is the number of
		* of parameter in the support region (per dimension ). */   
		unsigned long m_NumBSplineWeights;

    /** Variables for the internal mask */
    typename InternalMovingImageMaskType::Pointer      m_InternalMovingImageMask;
    typename MovingImageMaskInterpolatorType::Pointer  m_MovingImageMaskInterpolator;

    /** Some arrays for computing derivatives */
    mutable DerivativeType                        m_AlphaDerivatives;
    mutable DerivativeType                        m_ImageJacobian;
    mutable DerivativeType                        m_MaskJacobian;
    mutable ParameterIndexArrayType               m_NonZeroJacobian;

    /** Variables for the pdfs */
    mutable MarginalPDFType                       m_FixedImageMarginalPDF;
		mutable MarginalPDFType                       m_MovingImageMarginalPDF;
    typename JointPDFType::Pointer								m_JointPDF;
		typename JointPDFDerivativesType::Pointer			m_JointPDFDerivatives;
    mutable JointPDFRegionType                    m_JointPDFWindow;
    double m_MovingImageNormalizedMin;
		double m_FixedImageNormalizedMin;
    double m_FixedImageTrueMin;
		double m_FixedImageTrueMax;
		double m_MovingImageTrueMin;
		double m_MovingImageTrueMax;
		double m_FixedImageBinSize;
		double m_MovingImageBinSize;
    unsigned long m_NumberOfParameters;
    double m_FixedParzenTermToIndexOffset;
    double m_MovingParzenTermToIndexOffset;
    
    /** Kernels for computing Parzen histograms and derivatives. */
		typename KernelFunctionType::Pointer m_FixedKernel;
    typename KernelFunctionType::Pointer m_MovingKernel;
		typename KernelFunctionType::Pointer m_DerivativeMovingKernel;
				    
    /** Parameters for the soft gray value limiter */
    double m_SoftMaxLimit_a;
    double m_SoftMaxLimit_A;
    double m_SoftMinLimit_a;
    double m_SoftMinLimit_A;

    /** The minimum and maximum gray values that fit in the histogram */
    double m_MovingImageMinLimit;
    double m_MovingImageMaxLimit;
    double m_FixedImageMinLimit;
    double m_FixedImageMaxLimit;

    /** Compute image derivatives at a point.
     * If a BSplineInterpolationFunction is used, this class obtain
		 * image derivatives from the BSpline interpolator. Otherwise, 
		 * image derivatives are computed using central differencing. */
		virtual void ComputeImageDerivatives( 
      const MovingImageContinuousIndexType & cindex,
			ImageDerivativesType& gradient ) const;

    /** Compute the image value (and possibly derivative) at a transformed point.
     * Checks if the point lies within the moving image buffer.
     * If no gradient is wanted, set the gradient argument to 0. */
    virtual void EvaluateMovingImageValueAndDerivative( 
      const MovingImagePointType & mappedPoint,
      bool & sampleOk,
      double & movingImageValue,
      ImageDerivativesType * gradient) const;
				
 		/** Transform a point from FixedImage domain to MovingImage domain.
		 * This function also checks if mapped point is within support region. */
		virtual void TransformPoint( const FixedImagePointType& fixedImagePoint,
			MovingImagePointType& mappedPoint, bool& sampleWithinSupportRegion ) const;
    		
    /** Estimate value and spatial derivative of internal moving mask */
    virtual void EvaluateMovingMaskValueAndDerivative(
      const MovingImagePointType & point,
      double & value,
      MovingImageMaskDerivativeType & derivative) const;
  
    /** Estimate value of internal moving mask */
    virtual void EvaluateMovingMaskValue(
      const MovingImagePointType & point,
      double & value ) const;

    /** Compute the parzen values given an image value and a starting histogram index
     * Compute the values at (parzenWindowIndex - parzenWindowTerm + k) for 
     * k = 0 ... kernelsize-1
     * Returns the values in a ParzenValueContainer, which is supposed to have
     * the right size already **/
    void ComputeParzenValues(
      double parzenWindowTerm, int parzenWindowIndex,
      const KernelFunctionType * kernel, ParzenValueContainerType & parzenValues) const;
    
    /** Update the joint PDF with a pixel pair; on demand also updates the 
     * pdf derivatives */
    virtual void UpdateJointPDFAndDerivatives(
      double fixedImageValue, double movingImageValue, double movingMaskValue,
      bool updateDerivatives) const;

    /** Update the pdf derivatives
     * adds -grad_jac[mu]*factor_a + mask_jac[mu]*factor_b to the bin 
     * with index [ mu, pdfIndex[0], pdfIndex[1] ] for all mu.
     * This function should only be called from UpdateJointPDFAndDerivatives */
    void UpdateJointPDFDerivatives(
      const JointPDFIndexType & pdfIndex, double factor_a, double factor_b) const;

    /** Computes the innerproduct of transform jacobian with moving image gradient
     * and transform jacobian with the derivative of the movingMask
     * The results are stored in m_ImageJacobian and
     * m_MaskJacobian; m_NonZeroJacobian contains a list of 
     * parameter numbers that have a nonzero transformJacobian. If it's empty
     * it means that all parameter numbers have a nonzero jacobian. */
    virtual void ComputeTransformJacobianInnerProducts(
      const FixedImagePointType & fixedImagePoint, 
		  const ImageDerivativesType & movingImageGradientValue,
      const MovingImageMaskDerivativeType & movingMaskDerivative) const;

    /** Adds the MaskJacobian to the alpha derivative vector */
    virtual void UpdateAlphaDerivatives(void) const;	

    /** Check if enough samples have been found to compute a reliable 
     * estimate of the value/derivative; throws an exception if not */
    virtual void CheckNumberOfSamples(
      unsigned long wanted, unsigned long found, double sumOfMaskValues) const;

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
      const JointPDFType * jointPDF, MarginalPDFType & marginalPDF, unsigned int direction ) const;

    /** Functions called from Initialize, to split up that function a bit. */
    virtual void ComputeImageExtrema(
      double & fixedImageMin, double & fixedImageMax,
      double & movingImageMin, double & movingImageMax) const;   
    virtual void InitializeLimiter(void);
    virtual void InitializeHistograms(void);
    virtual void InitializeKernels(void);
    virtual void CheckForBSplineInterpolator(void);
    virtual void CheckForBSplineTransform(void);
    virtual void InitializeInternalMasks(void);

	private:
		
		/** The private constructor. */
		MattesMutualInformationImageToImageMetric3( const Self& );	// purposely not implemented
		/** The private copy constructor. */
		void operator=( const Self& );															// purposely not implemented
    				
		unsigned long m_NumberOfFixedHistogramBins;
    unsigned long m_NumberOfMovingHistogramBins;
    bool m_CheckNumberOfSamples;
    bool m_UseDifferentiableOverlap;
    bool m_HardLimitMovingGrayValues;
    bool m_SoftLimitMovingGrayValues;
    double m_FixedLimitRangeRatio;
    double m_MovingLimitRangeRatio;
    unsigned int m_FixedKernelBSplineOrder;
    unsigned int m_MovingKernelBSplineOrder;
		
	}; // end class MattesMutualInformationImageToImageMetric3

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkMattesMutualInformationImageToImageMetric3.hxx"
#endif

#endif // end #ifndef __itkMattesMutualInformationImageToImageMetric3_H__

