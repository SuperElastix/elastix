#ifndef __itkLinearScalingGroupwiseMetric_H__
#define __itkLinearScalingGroupwiseMetric_H__

#include "itkAdvancedImageToImageMetric.h"

#include "itkMetricSum.h"
#include "itkMetricAverage.h"
#include "itkMetricSquaredSum.h"
#include "itkMetricSquareRootSum.h"

#include "itkAATemplate.h"
#include "itkReducedAATemplate.h"
#include "itkGATemplate.h"
#include "itkHATemplate.h"
#include "itkMedianTemplate.h"


namespace itk
{
    
    template< class TFixedImage, class TMovingImage >
    class LinearScalingGroupwiseMetric : public AdvancedImageToImageMetric< TFixedImage, TMovingImage >
    {
        public:
        
            typedef LinearScalingGroupwiseMetric                    Self;
            typedef AdvancedImageToImageMetric<TFixedImage, TMovingImage >  Superclass;
            typedef SmartPointer< Self >                                        Pointer;
            typedef SmartPointer< const Self >                                  ConstPointer;
        
            itkTypeMacro(LinearScalingGroupwiseMetric, AdvancedImageToImageMetric );
        
        typedef typename Superclass::CoordinateRepresentationType    CoordinateRepresentationType;
        typedef typename Superclass::MovingImageType                 MovingImageType;
        typedef typename Superclass::MovingImagePixelType            MovingImagePixelType;
        typedef typename Superclass::MovingImageConstPointer         MovingImageConstPointer;
        typedef typename Superclass::FixedImageType                  FixedImageType;
        typedef typename FixedImageType::SizeType                    FixedImageSizeType;
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
        
        typedef itk::MetricCombiner        MetricCombinerType;
        typedef typename MetricCombinerType::Pointer                         MetricCombinerPointer;
        typedef itk::MetricSum              MetricSumType;
        typedef itk::MetricAverage          MetricAverageType;
        typedef itk::MetricSquaredSum       MetricSquaredSumType;
        typedef itk::MetricSquareRootSum    MetricSquareRootSumType;
        typedef itk::TemplateImage          TemplateImageType;
        typedef typename TemplateImageType::Pointer                     TemplateImagePointer;
        typedef itk::AATemplate             AATemplateType;
        typedef itk::ReducedAATemplate             ReducedAATemplateType;
        typedef itk::GATemplate            GATemplateType;
        typedef itk::HATemplate            HATemplateType;
        typedef itk::MedianTemplate        MedianTemplateType;
        
        void SetCombinationToSum( void );
        void SetCombinationToAverage( void );
        void SetCombinationToSqSum( void );
        void SetCombinationToSqRSum( void );
        
        
        void SetTemplateToAA( void );
        void SetTemplateToReducedAA( void );
        void SetTemplateToGA( void );
        void SetTemplateToHA( void );
        void SetTemplateToMedian( void );

        
            itkStaticConstMacro( FixedImageDimension, unsigned int, FixedImageType::ImageDimension );
        
            itkStaticConstMacro( MovingImageDimension, unsigned int, MovingImageType::ImageDimension );
        
            itkStaticConstMacro( ReducedFixedImageDimension, unsigned int, FixedImageType::ImageDimension - 1 );
        
        itkStaticConstMacro( ReducedMovingImageDimension, unsigned int, MovingImageType::ImageDimension - 1 );
        
        itkSetMacro( SampleLastDimensionRandomly, bool );
        itkSetMacro( NumSamplesLastDimension, unsigned int );
        itkSetMacro( SubtractMean, bool );
        itkSetMacro( TransformIsStackTransform, bool );
        itkSetMacro( GridSize, FixedImageSizeType );
        itkSetMacro( TemplateImage, TemplateImagePointer );
        itkSetMacro( MetricCombiner, MetricCombinerPointer );

        
            MeasureType GetValue( const ParametersType & parameters ) const;
        
            void GetDerivative(const ParametersType & parameters, DerivativeType & Derivative ) const;
        
            void GetValueAndDerivative( const ParametersType & parameters, MeasureType & value, DerivativeType & derivative) const;
        
        virtual void GetValue(const ParametersType & parameters, std::vector<MeasureType> & values, bool & minimize) const = 0;
        
        virtual void GetValueAndDerivative( const ParametersType & parameters, std::vector<MeasureType> & values, std::vector<DerivativeType> & derivatives, bool & minimize ) const = 0;

        
            virtual void Initialize( void ) throw ( ExceptionObject );
        
        protected:
        
            LinearScalingGroupwiseMetric();
        
            virtual ~LinearScalingGroupwiseMetric();
        
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
        
        unsigned int m_NumSamplesLastDimension;
        bool m_SampleLastDimensionRandomly;
        bool m_SubtractMean;
        bool m_TransformIsStackTransform;
        FixedImageSizeType m_GridSize;
        
        void SampleRandom( const int n, const int m, std::vector< int > & numbers ) const;
        void EvaluateTransformJacobianInnerProduct(const TransformJacobianType & jacobian, const MovingImageDerivativeType & movingImageDerivative, DerivativeType & imageJacobian ) const;
        
        TemplateImagePointer m_TemplateImage;
        MetricCombinerPointer m_MetricCombiner;
        
        mutable std::vector<int> m_RandomList;

        private:
        
            LinearScalingGroupwiseMetric( const Self & ); // purposely not implemented
            void operator=( const Self & );
    };
    
}

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkLinearScalingGroupwiseMetric.hxx"
#endif

#endif // end #ifndef __itkLinearScalingGroupwiseMetric_H__
