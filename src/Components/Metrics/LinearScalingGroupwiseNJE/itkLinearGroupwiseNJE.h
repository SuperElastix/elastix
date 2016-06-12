#ifndef __itkLinearGroupwiseNJE_H__
#define __itkLinearGroupwiseNJE_H__

#include "itkLinearScalingGroupwiseMetric.h"

namespace itk
{
    
    template< class TFixedImage, class TMovingImage >
    class LinearGroupwiseNJE : public LinearScalingGroupwiseMetric< TFixedImage, TMovingImage >
    {
    public:
        
        typedef LinearGroupwiseNJE                          Self;
        typedef LinearScalingGroupwiseMetric< TFixedImage, TMovingImage > Superclass;
        typedef SmartPointer< Self >                                    Pointer;
        typedef SmartPointer< const Self >                              ConstPointer;
        
        itkNewMacro( Self) ;
        
        itkTypeMacro( LinearGroupwiseNJE, LinearScalingGroupwiseMetric );
        
        typedef typename Superclass::CoordinateRepresentationType    CoordinateRepresentationType;
        typedef typename Superclass::MovingImageType                 MovingImageType;
        typedef typename Superclass::MovingImagePixelType            MovingImagePixelType;
        typedef typename Superclass::MovingImageConstPointer         MovingImageConstPointer;
        typedef typename Superclass::FixedImageType                  FixedImageType;
        typedef typename Superclass::FixedImageConstPointer          FixedImageConstPointer;
        typedef typename Superclass::FixedImageRegionType            FixedImageRegionType;
        typedef typename Superclass::TransformType                   TransformType;
        typedef typename Superclass::TransformPointer                TransformPointer;
        typedef typename Superclass::InputPointType                  InputPointType;
        typedef typename Superclass::OutputPointType                 OutputPointType;
        typedef typename Superclass::TransformParametersType         TransformParametersType;
        typedef typename Superclass::TransformJacobianType           TransformJacobianType;
        typedef typename Superclass::InterpolatorType                InterpolatorType;
        typedef typename Superclass::InterpolatorPointer             InterpolatorPointer;
        typedef typename Superclass::RealType                        RealType;
        typedef typename Superclass::GradientPixelType               GradientPixelType;
        typedef typename Superclass::GradientImageType               GradientImageType;
        typedef typename Superclass::GradientImagePointer            GradientImagePointer;
        typedef typename Superclass::GradientImageFilterType         GradientImageFilterType;
        typedef typename Superclass::GradientImageFilterPointer      GradientImageFilterPointer;
        typedef typename Superclass::FixedImageMaskType              FixedImageMaskType;
        typedef typename Superclass::FixedImageMaskPointer           FixedImageMaskPointer;
        typedef typename Superclass::MovingImageMaskType             MovingImageMaskType;
        typedef typename Superclass::MovingImageMaskPointer          MovingImageMaskPointer;
        typedef typename Superclass::MeasureType                     MeasureType;
        typedef typename Superclass::DerivativeType                  DerivativeType;
        typedef typename Superclass::DerivativeValueType             DerivativeValueType;
        typedef typename Superclass::ParametersType                  ParametersType;
        typedef typename Superclass::FixedImagePixelType             FixedImagePixelType;
        typedef typename Superclass::MovingImageRegionType           MovingImageRegionType;
        typedef typename Superclass::ImageSamplerType                ImageSamplerType;
        typedef typename Superclass::ImageSamplerPointer             ImageSamplerPointer;
        typedef typename Superclass::ImageSampleContainerType        ImageSampleContainerType;
        typedef typename Superclass::ImageSampleContainerPointer     ImageSampleContainerPointer;
        typedef typename Superclass::FixedImageLimiterType           FixedImageLimiterType;
        typedef typename Superclass::MovingImageLimiterType          MovingImageLimiterType;
        typedef typename Superclass::FixedImageLimiterOutputType     FixedImageLimiterOutputType;
        typedef typename Superclass::MovingImageLimiterOutputType    MovingImageLimiterOutputType;
        typedef typename Superclass::MovingImageDerivativeScalesType MovingImageDerivativeScalesType;
        typedef typename Superclass::ThreaderType                    ThreaderType;
        typedef typename Superclass::ThreadInfoType                  ThreadInfoType;
        typedef typename Superclass::NumberOfParametersType          NumberOfParametersType;
        
        typedef itk::TemplateImage      TemplateImageType;
        typedef typename TemplateImageType::Pointer                 TemplateImagePointer;
        
        itkStaticConstMacro( FixedImageDimension, unsigned int, FixedImageType::ImageDimension );
        
        itkStaticConstMacro( MovingImageDimension, unsigned int, MovingImageType::ImageDimension );
        
        itkStaticConstMacro( ReducedFixedImageDimension, unsigned int, FixedImageType::ImageDimension - 1 );
        
        itkStaticConstMacro( ReducedMovingImageDimension, unsigned int, MovingImageType::ImageDimension - 1 );
        
        itkSetMacro( UseExplicitPDFDerivatives, bool );
        
        itkGetConstReferenceMacro( UseExplicitPDFDerivatives, bool );
        
        itkBooleanMacro( UseExplicitPDFDerivatives );
        
        itkSetMacro( UseDerivative, bool );
        
        itkGetConstMacro( UseDerivative, bool );

        
        itkSetClampMacro( FixedKernelBSplineOrder, unsigned int, 0, 3 );
        
        itkSetClampMacro( MovingKernelBSplineOrder, unsigned int, 0, 3 );
        
        itkSetClampMacro( NumberOfFixedHistogramBins, unsigned long, 4, NumericTraits< unsigned long >::max() );
        
        itkSetClampMacro( NumberOfMovingHistogramBins, unsigned long, 4, NumericTraits< unsigned long >::max() );
        
        itkGetConstMacro( FixedKernelBSplineOrder, unsigned int );
        
        itkGetConstMacro( MovingKernelBSplineOrder, unsigned int );
        
        itkGetMacro( NumberOfFixedHistogramBins, unsigned long );
        
        itkGetMacro( NumberOfMovingHistogramBins, unsigned long );
        
        itkSetMacro( MovingImageLimitRangeRatio, double );
        
        itkGetConstMacro( MovingImageLimitRangeRatio, double );
        
        itkSetMacro( FixedImageLimitRangeRatio, double );
        
        itkGetConstMacro( FixedImageLimitRangeRatio, double );
        

        virtual void Initialize( void ) throw ( ExceptionObject );
        
        virtual void InitializeKernels( void );
        
        virtual void InitializeVectors( void );
        
        virtual void InitializeHistograms( void );

        
        virtual void GetValue(const ParametersType & parameters, std::vector<MeasureType> & values, bool & minimize) const;
        
        virtual void GetValueAndDerivative( const ParametersType & parameters, std::vector<MeasureType> & values, std::vector<DerivativeType> & derivatives, bool & minimize) const;
        
        virtual void GetValueAndAnalyticalDerivative( const ParametersType & parameters, std::vector<MeasureType> & values, std::vector<DerivativeType> & derivatives ) const;
        
        virtual void GetValueAndAnalyticalDerivativeLowMemory( const ParametersType & parameters, std::vector<MeasureType> & values, std::vector<DerivativeType> & derivatives ) const;
        
        using LinearScalingGroupwiseMetric< TFixedImage, TMovingImage >::GetValueAndDerivative;
        using LinearScalingGroupwiseMetric< TFixedImage, TMovingImage >::GetValue;

        LinearGroupwiseNJE();
        
        virtual ~LinearGroupwiseNJE();
        
    protected:
        
        typedef typename Superclass::FixedImageIndexType                                            FixedImageIndexType;
        typedef typename Superclass::FixedImageIndexValueType                                       FixedImageIndexValueType;
        typedef typename FixedImageType::OffsetValueType                                            OffsetValueType;
        typedef typename Superclass::MovingImageIndexType                                           MovingImageIndexType;
        typedef typename Superclass::FixedImagePointType                                            FixedImagePointType;
        typedef typename Superclass::MovingImagePointType                                           MovingImagePointType;
        typedef typename itk::ContinuousIndex< CoordinateRepresentationType, FixedImageDimension >  FixedImageContinuousIndexType;
        typedef typename Superclass::MovingImageContinuousIndexType                                 MovingImageContinuousIndexType;
        typedef typename Superclass::BSplineInterpolatorType                                        BSplineInterpolatorType;
        typedef typename Superclass::MovingImageDerivativeType                                      MovingImageDerivativeType;
        typedef typename Superclass::CentralDifferenceGradientFilterType                            CentralDifferenceGradientFilterType;
        typedef typename Superclass::NonZeroJacobianIndicesType                                     NonZeroJacobianIndicesType;
        
        typedef double                                       PDFValueType;
        typedef float                                        PDFDerivativeValueType;
        typedef Array< PDFValueType >                        MarginalPDFType;
        typedef Image< PDFValueType, 2 >                     JointPDFType;
        typedef typename JointPDFType::Pointer               JointPDFPointer;
        typedef Image< PDFDerivativeValueType, 3 >           JointPDFDerivativesType;
        typedef typename JointPDFDerivativesType::Pointer    JointPDFDerivativesPointer;
        typedef JointPDFType::IndexType                      JointPDFIndexType;
        typedef JointPDFType::RegionType                     JointPDFRegionType;
        typedef JointPDFType::SizeType                       JointPDFSizeType;
        typedef JointPDFDerivativesType::IndexType           JointPDFDerivativesIndexType;
        typedef JointPDFDerivativesType::RegionType          JointPDFDerivativesRegionType;
        typedef JointPDFDerivativesType::SizeType            JointPDFDerivativesSizeType;
        typedef Array< PDFValueType >                        ParzenValueContainerType;
        
        typedef KernelFunctionBase< PDFValueType >   KernelFunctionType;
        typedef typename KernelFunctionType::Pointer KernelFunctionPointer;
        
        typedef double                                PRatioType;
        typedef Array2D< PRatioType >                 PRatioArrayType;
        
        
        struct LinearGroupwiseNJEMultiThreaderParameterType
        {
            Self * m_Metric;
        };
        
        LinearGroupwiseNJEMultiThreaderParameterType m_LinearGroupwiseNJEThreaderParameters;
        
        struct LinearGroupwiseNJEGetValueAndDerivativePerThreadStruct
        {
            SizeValueType                st_NumberOfPixelsCounted;
            std::vector<SizeValueType>   st_NumberOfPixelsCountedVector;
            std::vector<JointPDFPointer> st_JointPDFs;
            std::vector<DerivativeType>  st_Derivatives;
        };
        
        itkPadStruct( ITK_CACHE_LINE_ALIGNMENT, LinearGroupwiseNJEGetValueAndDerivativePerThreadStruct, PaddedLinearGroupwiseNJEGetValueAndDerivativePerThreadStruct );
        itkAlignedTypedef( ITK_CACHE_LINE_ALIGNMENT, PaddedLinearGroupwiseNJEGetValueAndDerivativePerThreadStruct, AlignedLinearGroupwiseNJEGetValueAndDerivativePerThreadStruct );
        
        mutable AlignedLinearGroupwiseNJEGetValueAndDerivativePerThreadStruct *  m_LinearGroupwiseNJEGetValueAndDerivativePerThreadVariables;
        mutable ThreadIdType                                                                    m_LinearGroupwiseNJEGetValueAndDerivativePerThreadVariablesSize;

        bool            m_UseExplicitPDFDerivatives;
        bool          	m_UseDerivative;
        unsigned int  	m_FixedKernelBSplineOrder;
        unsigned int  	m_MovingKernelBSplineOrder;
        unsigned long 	m_NumberOfFixedHistogramBins;
        unsigned long 	m_NumberOfMovingHistogramBins;
        KernelFunctionPointer m_FixedKernel;
        KernelFunctionPointer m_MovingKernel;
        KernelFunctionPointer m_DerivativeMovingKernel;
        
        mutable JointPDFRegionType    				m_JointPDFWindow;
        
        double  m_FixedParzenTermToIndexOffset;
        double  m_MovingParzenTermToIndexOffset;
        
        std::vector<double>                         m_FixedImageTrueMins;
        std::vector<FixedImageLimiterOutputType>	m_FixedImageMinLimits;
        std::vector<double>                         m_FixedImageTrueMaxs;
        std::vector<FixedImageLimiterOutputType>	m_FixedImageMaxLimits;
        std::vector<double>           				m_FixedImageNormalizedMins;
        std::vector<double>           				m_FixedImageBinSizes;
        mutable std::vector<PRatioArrayType*>       m_PRatioArray;
        //mutable std::vector<int>                    m_RandomList;
        mutable std::vector<unsigned long>          m_NumberOfPixelsCountedVector;

        double                        				m_MovingImageBinSize;
        double                        				m_MovingImageNormalizedMin;
        double                                      m_FixedImageLimitRangeRatio;
        double                                      m_MovingImageLimitRangeRatio;
        
        mutable std::vector<MarginalPDFType*>     	m_FixedImageMarginalPDFs;
        mutable std::vector<MarginalPDFType*>     	m_MovingImageMarginalPDFs;
        std::vector<JointPDFPointer>   				m_JointPDFs;
        std::vector<JointPDFDerivativesPointer>		m_JointPDFDerivatives;
        
        virtual void NormalizeJointPDF( JointPDFType * pdf, const double & factor ) const;

        virtual void ComputeMarginalPDF( const JointPDFType * jointPDF , MarginalPDFType* marginalPDF, const unsigned int & direction ) const;
        
        virtual void ComputeLogMarginalPDF(MarginalPDFType* marginalPDF) const;

        virtual void EvaluateParzenValues( double parzenWindowTerm, OffsetValueType parzenWindowIndex, const KernelFunctionType * kernel, ParzenValueContainerType & parzenValues ) const;
        
        
        void ComputePDFs( const ParametersType & parameters ) const;
        
        static ITK_THREAD_RETURN_TYPE ComputePDFsThreaderCallback( void * arg );
        
        void LaunchComputePDFsThreaderCallback( void ) const;
        
        void InitializeThreadingParameters( void ) const;
        
        void AfterThreadedComputePDFs( void ) const;
        
        void ComputePDFsSingleThreaded( const ParametersType & parameters ) const;
        
        void ThreadedComputePDFs( ThreadIdType threadId );
        
        
        void ComputePDFsAndPDFDerivatives( const ParametersType & parameters ) const;

        void ComputePDFsAndPDFDerivativesSingleThreaded( const ParametersType & parameters ) const;

        
        void ComputeValueAndPRatioArray( double & nMI, unsigned int n ) const;
        
        void ComputeDerivativeLowMemory( std::vector<DerivativeType> & derivatives) const;
        
        static ITK_THREAD_RETURN_TYPE ComputeDerivativeLowMemoryThreaderCallback( void * arg );
        
        void AfterThreadedComputeDerivativeLowMemory( std::vector<DerivativeType> & derivatives ) const;

        void LaunchComputeDerivativeLowMemoryThreaderCallback( void ) const;
        
        void ComputeDerivativeLowMemorySingleThreaded( std::vector<DerivativeType> & derivatives) const;
        
        void ThreadedComputeDerivativeLowMemory( ThreadIdType threadId );
        
        
        void UpdateJointPDFAndDerivatives(const std::vector<double> & fixedImageValue, const std::vector<DerivativeType> * imageJacobian, const std::vector<NonZeroJacobianIndicesType> * nzji, JointPDFType * jointPDF, const unsigned int n, const std::vector<unsigned int> & positions ) const;
        
        void UpdateJointPDFDerivatives( const std::vector<double> & fixedImageValue, const double & movingImageValue, const JointPDFIndexType & pdfIndex, double factor, double factorr, const std::vector<DerivativeType> * imageJacobian, const std::vector<NonZeroJacobianIndicesType> * nzji, const unsigned int n, const std::vector<unsigned int> & positions ) const;
        
        void UpdateDerivativeLowMemory(const std::vector<double> & fixedImageValue, const std::vector<DerivativeType> * imageJacobian, const std::vector<NonZeroJacobianIndicesType> * nzji, const unsigned int n, const std::vector<unsigned int> & positions, std::vector<DerivativeType> & derivatives) const;
        
    private:
        
        LinearGroupwiseNJE( const Self & ); // purposely not implemented
        void operator=( const Self & );
        
    };
    
}

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkLinearGroupwiseNJE.hxx"
#endif

#endif // end #ifndef __itkLinearGroupwiseNJE_H__
