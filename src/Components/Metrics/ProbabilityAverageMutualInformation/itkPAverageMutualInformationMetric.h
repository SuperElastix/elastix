#ifndef __itkPAverageMutualInformationMetric_H__
#define __itkPAverageMutualInformationMetric_H__

#include "itkArray2D.h"

namespace itk
{
    
    template< class TFixedImage, class TMovingImage >
    class PAverageMutualInformationMetric : public AdvancedImageToImageMetric< TFixedImage, TMovingImage >
    {
        public:
        
            typedef PAverageMutualInformationMetric                          Self;
            typedef AdvancedImageToImageMetric< TFixedImage, TMovingImage > Superclass;
            typedef SmartPointer< Self >                                    Pointer;
            typedef SmartPointer< const Self >                              ConstPointer;
        
            itkTypeMacro( PAverageMutualInformationMetric, AdvancedImageToImageMetric );
        
            typedef typename Superclass::CoordinateRepresentationType    CoordinateRepresentationType;
            typedef typename Superclass::MovingImageType                 MovingImageType;
            typedef typename Superclass::MovingImagePixelType            MovingImagePixelType;
            typedef typename Superclass::MovingImageConstPointer         MovingImageConstPointer;
            typedef typename Superclass::FixedImageType                  FixedImageType;
            typedef typename Superclass::FixedImageConstPointer          FixedImageConstPointer;
            typedef typename Superclass::FixedImageRegionType            FixedImageRegionType;
            typedef typename FixedImageRegionType::SizeType                                             FixedImageSizeType;
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
        
            itkStaticConstMacro( FixedImageDimension, unsigned int, FixedImageType::ImageDimension );
        
            itkStaticConstMacro( MovingImageDimension, unsigned int, MovingImageType::ImageDimension );
        
            itkStaticConstMacro( ReducedFixedImageDimension, unsigned int, FixedImageType::ImageDimension - 1 );
        
            itkStaticConstMacro( ReducedMovingImageDimension, unsigned int, MovingImageType::ImageDimension - 1 );
        
            itkSetMacro( GridSize, FixedImageSizeType );
        
            itkSetMacro( TransformIsStackTransform, bool );
        
            itkSetMacro( SubtractMean, bool );
                
            itkSetMacro( UseDerivative, bool );
        
            itkSetMacro( UseExplicitPDFDerivatives, bool );
        
            itkSetMacro( SampleLastDimensionRandomly, bool );
        
            itkSetMacro( NumSamplesLastDimension, unsigned int );
        
            itkSetClampMacro( FixedKernelBSplineOrder, unsigned int, 0, 3 );
        
            itkSetClampMacro( MovingKernelBSplineOrder, unsigned int, 0, 3 );
        
            itkSetClampMacro( NumberOfFixedHistogramBins, unsigned long, 2, NumericTraits< unsigned long >::max() );
        
            itkSetClampMacro( NumberOfMovingHistogramBins, unsigned long, 2, NumericTraits< unsigned long >::max() );
        
            itkGetConstMacro( UseDerivative, bool );
        
            itkGetConstReferenceMacro( UseExplicitPDFDerivatives, bool );
        
            itkGetConstMacro( SampleLastDimensionRandomly, bool );
        
            itkGetConstMacro( NumSamplesLastDimension, unsigned int );
        
            itkGetConstMacro( FixedKernelBSplineOrder, unsigned int );
        
            itkGetConstMacro( MovingKernelBSplineOrder, unsigned int );
        
            itkGetMacro( NumberOfFixedHistogramBins, unsigned long );

            itkGetMacro( NumberOfMovingHistogramBins, unsigned long );
        
            itkBooleanMacro( UseExplicitPDFDerivatives );
        
            /*******************/
            //Functions
            /*******************/
        
            virtual void Initialize( void ) throw ( ExceptionObject );
        
            virtual void InitializeVectors( void );
        
            virtual void InitializeKernels( void );
        
        protected:
        
            PAverageMutualInformationMetric();
        
            virtual ~PAverageMutualInformationMetric();
        
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
        
            FixedImageSizeType m_GridSize;
        
            bool            m_TransformIsStackTransform;
            bool            m_SubtractMean;
            bool          	m_UseDerivative;
            bool            m_UseExplicitPDFDerivatives;
            bool            m_SampleLastDimensionRandomly;
            unsigned int    m_NumSamplesLastDimension;
        
            unsigned int  	m_FixedKernelBSplineOrder;
            unsigned int  	m_MovingKernelBSplineOrder;
            unsigned long 	m_NumberOfFixedHistogramBins;
            unsigned long 	m_NumberOfMovingHistogramBins;
        
            std::vector<FixedImagePixelType>            m_FixedImageTrueMins;
            std::vector<FixedImageLimiterOutputType>	m_FixedImageMinLimits;
            std::vector<FixedImagePixelType>            m_FixedImageTrueMaxs;
            std::vector<FixedImageLimiterOutputType>	m_FixedImageMaxLimits;
            std::vector<double>           				m_FixedImageNormalizedMins;
            std::vector<double>           				m_FixedImageBinSizes;
        
            std::vector<MovingImagePixelType>           m_MovingImageTrueMins;
            std::vector<MovingImageLimiterOutputType>	m_MovingImageMinLimits;
            std::vector<MovingImagePixelType>           m_MovingImageTrueMaxs;
            std::vector<MovingImageLimiterOutputType>	m_MovingImageMaxLimits;
            std::vector<double>           				m_MovingImageNormalizedMins;
            std::vector<double>           				m_MovingImageBinSizes;
        
            mutable double                              m_Alpha;

            mutable std::vector<MarginalPDFType*>     	m_FixedImageMarginalPDFs;
            mutable std::vector<MarginalPDFType*>     	m_MovingImageMarginalPDFs;
            std::vector<JointPDFPointer>   				m_JointPDFs;
            std::vector<JointPDFDerivativesPointer>		m_JointPDFDerivatives;
            mutable JointPDFRegionType    				m_JointPDFWindow;
        
            mutable std::vector<double>                 m_Values;
            mutable std::vector<unsigned int>           m_RandomList;
        
            KernelFunctionPointer m_FixedKernel;
            KernelFunctionPointer m_MovingKernel;
            KernelFunctionPointer m_DerivativeMovingKernel;
        
            double  m_FixedParzenTermToIndexOffset;
            double  m_MovingParzenTermToIndexOffset;
        
            typedef double                                PRatioType;
            typedef Array2D< PRatioType >                 PRatioArrayType;
            mutable std::vector<PRatioArrayType*>         m_PRatioArray;

            /*******************/
            //Functions
            /*******************/
        
            virtual void SampleRandom( const int n, const int m, std::vector< unsigned int > & numbers ) const;
        
            virtual void NormalizeJointPDF( JointPDFType * pdf, const double & factor ) const;

            virtual void ComputeMarginalPDF( const JointPDFType * jointPDF , MarginalPDFType* marginalPDF, const unsigned int & direction ) const;
        
            virtual void EvaluateParzenValues( double parzenWindowTerm, OffsetValueType parzenWindowIndex, const KernelFunctionType * kernel, ParzenValueContainerType & parzenValues ) const;
        
            virtual void ComputeValueAndPRatioArray( double & MI, unsigned int n ) const;
        
        private:
        
            PAverageMutualInformationMetric( const Self & ); // purposely not implemented
            void operator=( const Self & );                 // purposely not implemented

        
    };
    
}

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkPAverageMutualInformationMetric.hxx"
#endif

#endif // end #ifndef __itkPAverageMutualInformationMetric_H__
