#ifndef __itkLinearGroupwiseMSD_H__
#define __itkLinearGroupwiseMSD_H__

#include "itkLinearScalingGroupwiseMetric.h"

namespace itk
{
    
    template< class TFixedImage, class TMovingImage >
    class LinearGroupwiseMSD : public LinearScalingGroupwiseMetric< TFixedImage, TMovingImage >
    {
    public:
        
        typedef LinearGroupwiseMSD                          Self;
        typedef LinearScalingGroupwiseMetric< TFixedImage, TMovingImage > Superclass;
        typedef SmartPointer< Self >                                    Pointer;
        typedef SmartPointer< const Self >                              ConstPointer;
        
        itkNewMacro( Self) ;
        
        itkTypeMacro( LinearGroupwiseMSD, LinearScalingGroupwiseMetric );
        
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
        
        virtual void Initialize( void ) throw ( ExceptionObject );
        
        virtual void GetValue(const ParametersType & parameters, std::vector<MeasureType> & values , bool & minimize) const;
                
        virtual void GetValueAndDerivative( const ParametersType & parameters, std::vector<MeasureType> & values, std::vector<DerivativeType> & derivatives, bool & minimize) const;
        
        virtual void GetValueAndDerivativeSingleThreaded( const ParametersType & parameters, std::vector<MeasureType> & values, std::vector<DerivativeType> & derivatives ) const;
        
        virtual void InitializeThreadingParameters( ) const;

        virtual void ThreadedGetValueAndDerivative( ThreadIdType threadId );
        
        virtual void AfterThreadedGetValueAndDerivative(std::vector<MeasureType> & values, std::vector<DerivativeType> & derivatives ) const;

        LinearGroupwiseMSD();
        
        virtual ~LinearGroupwiseMSD();
        
    protected:
        
        typedef typename Superclass::FixedImageIndexType      FixedImageIndexType;
        typedef typename Superclass::FixedImageIndexValueType FixedImageIndexValueType;
        typedef typename Superclass::MovingImageIndexType     MovingImageIndexType;
        typedef typename Superclass::FixedImagePointType      FixedImagePointType;
        typedef typename itk::ContinuousIndex< CoordinateRepresentationType, FixedImageDimension >
        FixedImageContinuousIndexType;
        typedef typename Superclass::MovingImagePointType                MovingImagePointType;
        typedef typename Superclass::MovingImageContinuousIndexType      MovingImageContinuousIndexType;
        typedef typename Superclass::BSplineInterpolatorType             BSplineInterpolatorType;
        typedef typename Superclass::CentralDifferenceGradientFilterType CentralDifferenceGradientFilterType;
        typedef typename Superclass::MovingImageDerivativeType           MovingImageDerivativeType;
        typedef typename Superclass::NonZeroJacobianIndicesType          NonZeroJacobianIndicesType;
        
        struct LinearGroupwiseMSDMultiThreaderParameterType
        {
            Self * m_Metric;
        };
        
        LinearGroupwiseMSDMultiThreaderParameterType m_LinearGroupwiseMSDThreaderParameters;
        
        struct LinearGroupwiseMSDGetValueAndDerivativePerThreadStruct
        {
            SizeValueType   st_NumberOfPixelsCounted;
            std::vector<SizeValueType>   st_NumberOfPixelsCountedVector;
            std::vector<double> st_Values;
            std::vector<DerivativeType> st_Derivatives;
        };
        
        itkPadStruct( ITK_CACHE_LINE_ALIGNMENT, LinearGroupwiseMSDGetValueAndDerivativePerThreadStruct,
                     PaddedLinearGroupwiseMSDGetValueAndDerivativePerThreadStruct );
        itkAlignedTypedef( ITK_CACHE_LINE_ALIGNMENT, PaddedLinearGroupwiseMSDGetValueAndDerivativePerThreadStruct,
                          AlignedLinearGroupwiseMSDGetValueAndDerivativePerThreadStruct );
        mutable AlignedLinearGroupwiseMSDGetValueAndDerivativePerThreadStruct * m_LinearGroupwiseMSDGetValueAndDerivativePerThreadVariables;
        mutable ThreadIdType                                                       m_LinearGroupwiseMSDGetValueAndDerivativePerThreadVariablesSize;
        
        mutable std::vector<unsigned long>                    m_NumberOfPixelsCountedVector;

    private:
        
        LinearGroupwiseMSD( const Self & ); // purposely not implemented
        void operator=( const Self & );
        
        double m_InitialVariance;
    };
    
}

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkLinearGroupwiseMSD.hxx"
#endif

#endif // end #ifndef __itkLinearGroupwiseMSD_H__
