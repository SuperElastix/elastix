#ifndef __itkMCPANMI_H__
#define __itkMCPANMI_H__

#include "itkMCPAverageNormalizedMutualInformationMetric.h"

#include "itkArray2D.h"

namespace itk
{
    
    template< class TFixedImage, class TMovingImage >
    class MCPANMI : public MCPAverageNormalizedMutualInformationMetric< TFixedImage, TMovingImage >
    {
        public:
        
            typedef MCPANMI                    Self;
            typedef MCPAverageNormalizedMutualInformationMetric<TFixedImage, TMovingImage >  Superclass;
            typedef SmartPointer< Self >                                        Pointer;
            typedef SmartPointer< const Self >                                  ConstPointer;
        
            itkNewMacro( Self) ;
        
            itkTypeMacro(MCPANMI, MCPAverageNormalizedMutualInformationMetric );
        
            typedef typename Superclass::CoordinateRepresentationType       CoordinateRepresentationType;
            typedef typename Superclass::MovingImageType                    MovingImageType;
            typedef typename Superclass::MovingImagePixelType               MovingImagePixelType;
            typedef typename Superclass::MovingImageConstPointer            MovingImageConstPointer;
            typedef typename Superclass::FixedImageType                     FixedImageType;
            typedef typename Superclass::FixedImageConstPointer             FixedImageConstPointer;
            typedef typename Superclass::FixedImageRegionType               FixedImageRegionType;
            typedef typename Superclass::TransformType                      TransformType;
            typedef typename Superclass::TransformPointer                   TransformPointer;
            typedef typename Superclass::InputPointType                     InputPointType;
            typedef typename Superclass::OutputPointType                    OutputPointType;
            typedef typename Superclass::TransformParametersType            TransformParametersType;
            typedef typename Superclass::TransformJacobianType              TransformJacobianType;
            typedef typename Superclass::NumberOfParametersType             NumberOfParametersType;
            typedef typename Superclass::InterpolatorType                   InterpolatorType;
            typedef typename Superclass::InterpolatorPointer                InterpolatorPointer;
            typedef typename Superclass::RealType                           RealType;
            typedef typename Superclass::GradientPixelType                  GradientPixelType;
            typedef typename Superclass::GradientImageType                  GradientImageType;
            typedef typename Superclass::GradientImagePointer               GradientImagePointer;
            typedef typename Superclass::GradientImageFilterType            GradientImageFilterType;
            typedef typename Superclass::GradientImageFilterPointer         GradientImageFilterPointer;
            typedef typename Superclass::FixedImageMaskType                 FixedImageMaskType;
            typedef typename Superclass::FixedImageMaskPointer              FixedImageMaskPointer;
            typedef typename Superclass::MovingImageMaskType                MovingImageMaskType;
            typedef typename Superclass::MovingImageMaskPointer             MovingImageMaskPointer;
            typedef typename Superclass::MeasureType                        MeasureType;
            typedef typename Superclass::DerivativeType                     DerivativeType;
            typedef typename Superclass::DerivativeValueType                DerivativeValueType;
            typedef typename Superclass::ParametersType                     ParametersType;
            typedef typename Superclass::FixedImagePixelType                FixedImagePixelType;
            typedef typename Superclass::MovingImageRegionType              MovingImageRegionType;
            typedef typename Superclass::ImageSamplerType                   ImageSamplerType;
            typedef typename Superclass::ImageSamplerPointer                ImageSamplerPointer;
            typedef typename Superclass::ImageSampleContainerType           ImageSampleContainerType;
            typedef typename Superclass::ImageSampleContainerPointer        ImageSampleContainerPointer;
            typedef typename Superclass::FixedImageLimiterType              FixedImageLimiterType;
            typedef typename Superclass::MovingImageLimiterType             MovingImageLimiterType;
            typedef typename Superclass::FixedImageLimiterOutputType        FixedImageLimiterOutputType;
            typedef typename Superclass::MovingImageLimiterOutputType       MovingImageLimiterOutputType;
            typedef typename Superclass::MovingImageDerivativeScalesType    MovingImageDerivativeScalesType;
        
            typedef typename Superclass::ThreaderType   ThreaderType;
            typedef typename Superclass::ThreadInfoType ThreadInfoType;
        
        
            itkStaticConstMacro( FixedImageDimension, unsigned int, FixedImageType::ImageDimension );
        
            itkStaticConstMacro( MovingImageDimension, unsigned int, MovingImageType::ImageDimension );
        
            itkStaticConstMacro( ReducedFixedImageDimension, unsigned int, FixedImageType::ImageDimension - 1 );
        
            itkStaticConstMacro( ReducedMovingImageDimension, unsigned int, MovingImageType::ImageDimension - 1 );
        
            /*******************/
            //Functions
            /*******************/
        
            MeasureType GetValue( const ParametersType & parameters ) const;
        
            void GetDerivative(const ParametersType & parameters, DerivativeType & Derivative ) const;
        
            void GetValueAndDerivative( const ParametersType & parameters, MeasureType & value, DerivativeType & derivative ) const;
        
            void GetValueAndAnalyticalDerivative( const ParametersType & parameters, MeasureType & value, DerivativeType & derivative ) const;
        
            void GetValueAndAnalyticalDerivativeLowMemory( const ParametersType & parameters, MeasureType & value, DerivativeType & derivative ) const;
        
            void Initialize( void ) throw ( ExceptionObject );
        
            void FillVectors( void );
        
            void InitializeHistograms( void );
        
            void ComputePDFs( const ParametersType & parameters ) const;
        
            static ITK_THREAD_RETURN_TYPE ComputePDFsThreaderCallback( void * arg );
        
            void LaunchComputePDFsThreaderCallback( void ) const;
        
            void InitializeThreadingParameters( void ) const;
        
            void AfterThreadedComputePDFs( void ) const;
        
            void ComputePDFsSingleThreaded( const ParametersType & parameters ) const;
        
            void ThreadedComputePDFs( ThreadIdType threadId );
        
            void ComputePDFsAndPDFDerivatives( const ParametersType & parameters ) const;
                
            void ComputePDFsAndPDFDerivativesSingleThreaded( const ParametersType & parameters ) const;
        
            void ComputeDerivativeLowMemory( DerivativeType & derivative) const;
        
            static ITK_THREAD_RETURN_TYPE ComputeDerivativeLowMemoryThreaderCallback( void * arg );
        
            void LaunchComputeDerivativeLowMemoryThreaderCallback( void ) const;
        
            void AfterThreadedComputeDerivativeLowMemory( DerivativeType & derivative ) const;
        
            void ComputeDerivativeLowMemorySingleThreaded( DerivativeType & derivative) const;
        
            void ThreadedComputeDerivativeLowMemory( ThreadIdType threadId );

        
        protected:
        
            MCPANMI();
        
            virtual ~MCPANMI();
        
            typedef typename Superclass::FixedImageIndexType                 FixedImageIndexType;
            typedef typename Superclass::FixedImageIndexValueType            FixedImageIndexValueType;
            typedef typename Superclass::MovingImageIndexType                MovingImageIndexType;
            typedef typename Superclass::FixedImagePointType                 FixedImagePointType;
            typedef typename Superclass::MovingImagePointType                MovingImagePointType;
            typedef typename Superclass::MovingImageContinuousIndexType      MovingImageContinuousIndexType;
            typedef typename Superclass::BSplineInterpolatorType             BSplineInterpolatorType;
            typedef typename Superclass::CentralDifferenceGradientFilterType CentralDifferenceGradientFilterType;
            typedef typename Superclass::MovingImageDerivativeType           MovingImageDerivativeType;
            typedef typename Superclass::PDFValueType                        PDFValueType;
            typedef typename Superclass::PDFDerivativeValueType              PDFDerivativeValueType;
            typedef typename Superclass::MarginalPDFType                     MarginalPDFType;
            typedef typename Superclass::JointPDFType                        JointPDFType;
            typedef typename JointPDFType::Pointer               JointPDFPointer;

            typedef typename Superclass::JointPDFDerivativesType             JointPDFDerivativesType;
            typedef typename Superclass::JointPDFIndexType                   JointPDFIndexType;
            typedef typename Superclass::JointPDFRegionType                  JointPDFRegionType;
            typedef typename Superclass::JointPDFSizeType                    JointPDFSizeType;
            typedef typename Superclass::JointPDFDerivativesIndexType        JointPDFDerivativesIndexType;
            typedef typename Superclass::JointPDFDerivativesRegionType       JointPDFDerivativesRegionType;
            typedef typename Superclass::JointPDFDerivativesSizeType         JointPDFDerivativesSizeType;
            typedef typename Superclass::ParzenValueContainerType            ParzenValueContainerType;
            typedef typename Superclass::KernelFunctionType                  KernelFunctionType;
            typedef typename Superclass::NonZeroJacobianIndicesType          NonZeroJacobianIndicesType;
        
            typedef typename Superclass::PRatioType                                  PRatioType;
            typedef typename Superclass::PRatioArrayType                             PRatioArrayType;
        
            typedef typename itk::ContinuousIndex< CoordinateRepresentationType, FixedImageDimension > FixedImageContinuousIndexType;
        
        struct MCPANMIMultiThreaderParameterType
        {
            Self * m_Metric;
        };
        
        MCPANMIMultiThreaderParameterType m_MCPANMIThreaderParameters;
        
        struct MCPANMIGetValueAndDerivativePerThreadStruct
        {
            SizeValueType                st_NumberOfPixelsCounted;
            std::vector<SizeValueType>   st_NumberOfPixelsCountedVector;
            std::vector<JointPDFPointer> st_JointPDFs;
        };
        
        itkPadStruct( ITK_CACHE_LINE_ALIGNMENT, MCPANMIGetValueAndDerivativePerThreadStruct, PaddedMCPANMIGetValueAndDerivativePerThreadStruct );
        itkAlignedTypedef( ITK_CACHE_LINE_ALIGNMENT, PaddedMCPANMIGetValueAndDerivativePerThreadStruct, AlignedMCPANMIGetValueAndDerivativePerThreadStruct );
        
        mutable AlignedMCPANMIGetValueAndDerivativePerThreadStruct *  m_MCPANMIGetValueAndDerivativePerThreadVariables;
        mutable ThreadIdType                                                                    m_MCPANMIGetValueAndDerivativePerThreadVariablesSize;
                
            /*******************/
            //Functions
            /*******************/
        
        void UpdateJointPDFAndDerivatives(const std::vector<double> & fixedImageValue, const std::vector<JointPDFPointer> * jointPDFs, const std::vector<DerivativeType> * imageJacobian, const std::vector<NonZeroJacobianIndicesType> * nzji, const unsigned int n, const std::vector<unsigned int> & positions ) const;
        
            void UpdateJointPDFDerivatives( const JointPDFIndexType & pdfIndex, double factor, double factorr, const DerivativeType & imageJacobianFix, const NonZeroJacobianIndicesType & nzjiFix, const DerivativeType & imageJacobianMov, const NonZeroJacobianIndicesType & nzjiMov, const unsigned int o, const std::vector<unsigned int> & positions, const unsigned int n) const;
        
            void UpdateDerivativeLowMemory(const std::vector<double> & fixedImageValue, const std::vector<DerivativeType> * imageJacobian, const std::vector<NonZeroJacobianIndicesType> * nzji, const unsigned int n, const std::vector<unsigned int> & positions, DerivativeType & derivative) const;

        private:
        
            MCPANMI( const Self & ); // purposely not implemented
            void operator=( const Self & );

    };
    
}

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkMCPANMI.hxx"
#endif

#endif // end #ifndef __itkMCPANMI_H__
