#ifndef __elxProbabilityAverageNormalizedMutualInformationMetric_HXX__
#define __elxProbabilityAverageNormalizedMutualInformationMetric_HXX__

#include "elxProbabilityAverageNormalizedMutualInformationMetric.h"

#include "itkTimeProbe.h"


namespace elastix
{
    /**
     * ****************** Constructor ***********************
     */
    
    template< class TElastix >
    ProbabilityAverageNormalizedMutualInformationMetric< TElastix >
    ::ProbabilityAverageNormalizedMutualInformationMetric()
    {
        
    }
    
    /**
     * ******************* Initialize ***********************
     */
    
    template< class TElastix >
    void
    ProbabilityAverageNormalizedMutualInformationMetric< TElastix >
    ::Initialize( void ) throw ( itk::ExceptionObject )
    {
        itk::TimeProbe timer;
        timer.Start();
        this->Superclass1::Initialize();
        timer.Stop();
        elxout << "Initialization of ProbabilityAverageNormalizedMutualInformation metric took: "
        << static_cast< long >( timer.GetMean() * 1000 ) << " ms." << std::endl;

    }
        
    /**
     * ***************** BeforeRegistration ***********************
     */
        
    template< class TElastix >
    void
    ProbabilityAverageNormalizedMutualInformationMetric< TElastix >
    ::BeforeRegistration( void )
    {
        
        bool useFastAndLowMemoryVersion = false;
        this->GetConfiguration()->ReadParameter( useFastAndLowMemoryVersion, "UseFastAndLowMemoryVersion", this->GetComponentLabel(), 0, 0 );
        this->SetUseExplicitPDFDerivatives( !useFastAndLowMemoryVersion );
        
        this->SetUseMovingImageDerivativeScales( false );
        MovingImageDerivativeScalesType movingImageDerivativeScales;
        movingImageDerivativeScales.Fill( 1.0 );
        bool usescales = true;
        for( unsigned int i = 0; i < MovingImageDimension; ++i )
        {
            usescales = usescales && this->GetConfiguration()->ReadParameter( movingImageDerivativeScales[ i ], "MovingImageDerivativeScales", this->GetComponentLabel(), i, -1, true );
        }
        if( usescales )
        {
            this->SetUseMovingImageDerivativeScales( true );
            this->SetMovingImageDerivativeScales( movingImageDerivativeScales );
            elxout << "Multiplying moving image derivatives by: " << movingImageDerivativeScales << std::endl;
        }
        
    }

    /**
     * ***************** BeforeEachResolution ***********************
     */
        
    template< class TElastix >
    void
    ProbabilityAverageNormalizedMutualInformationMetric< TElastix >
    ::BeforeEachResolution( void )
    {
        unsigned int level = ( this->m_Registration->GetAsITKBaseType() )->GetCurrentLevel();
        
        bool useRandomSampling = false;
        this->GetConfiguration()->ReadParameter( useRandomSampling, "SampleLastDimensionRandomly", this->GetComponentLabel(), level, 0 );
        this->SetSampleLastDimensionRandomly( useRandomSampling );

        int numSamplesLastDimension = 3;
        this->GetConfiguration()->ReadParameter( numSamplesLastDimension, "NumSamplesLastDimension", this->GetComponentLabel(), level, 0 );
        this->SetNumSamplesLastDimension( numSamplesLastDimension );
        
        bool subtractMean = false;
        this->GetConfiguration()->ReadParameter( subtractMean, "SubtractMean", this->GetComponentLabel(), 0, 0 );
        this->SetSubtractMean( subtractMean );
        
        unsigned int numberOfHistogramBins = 32;
        this->GetConfiguration()->ReadParameter( numberOfHistogramBins, "NumberOfHistogramBins", this->GetComponentLabel(), level, 0 );
        this->SetNumberOfFixedHistogramBins( numberOfHistogramBins );
        this->SetNumberOfMovingHistogramBins( numberOfHistogramBins );
        
        unsigned int numberOfFixedHistogramBins  = numberOfHistogramBins;
        unsigned int numberOfMovingHistogramBins = numberOfHistogramBins;
        this->GetConfiguration()->ReadParameter( numberOfFixedHistogramBins, "NumberOfFixedHistogramBins", this->GetComponentLabel(), level, 0 );
        this->GetConfiguration()->ReadParameter( numberOfMovingHistogramBins, "NumberOfMovingHistogramBins", this->GetComponentLabel(), level, 0 );
        this->SetNumberOfFixedHistogramBins( numberOfFixedHistogramBins );
        this->SetNumberOfMovingHistogramBins( numberOfMovingHistogramBins );
        
        double fixedLimitRangeRatio  = 0.01;
        double movingLimitRangeRatio = 0.01;
        this->GetConfiguration()->ReadParameter( fixedLimitRangeRatio, "FixedLimitRangeRatio", this->GetComponentLabel(), level, 0 );
        this->GetConfiguration()->ReadParameter( movingLimitRangeRatio, "MovingLimitRangeRatio", this->GetComponentLabel(), level, 0 );
        this->SetFixedLimitRangeRatio( fixedLimitRangeRatio );
        this->SetMovingLimitRangeRatio( movingLimitRangeRatio );
        this->SetFixedImageLimitRangeRatio( fixedLimitRangeRatio );
        this->SetMovingImageLimitRangeRatio( movingLimitRangeRatio );
        
        unsigned int fixedKernelBSplineOrder  = 3;
        unsigned int movingKernelBSplineOrder = 3;
        this->GetConfiguration()->ReadParameter( fixedKernelBSplineOrder, "FixedKernelBSplineOrder", this->GetComponentLabel(), level, 0 );
        this->GetConfiguration()->ReadParameter( movingKernelBSplineOrder, "MovingKernelBSplineOrder", this->GetComponentLabel(), level, 0 );
        this->SetFixedKernelBSplineOrder( fixedKernelBSplineOrder );
        this->SetMovingKernelBSplineOrder( movingKernelBSplineOrder );

        CombinationTransformType * testPtr1 = dynamic_cast< CombinationTransformType * >( this->GetElastix()->GetElxTransformBase() );
        
        if( testPtr1 )
        {
            BSplineTransformBaseType * testPtr2 = dynamic_cast< BSplineTransformBaseType * >(testPtr1->GetCurrentTransform() );
            
            if( testPtr2 )
            {
                this->SetGridSize( testPtr2->GetGridRegion().GetSize() );
            }
            else
            {
                StackTransformType * testPtr3 = dynamic_cast< StackTransformType * >(testPtr1->GetCurrentTransform() );
                if( testPtr3 )
                {
                    this->SetTransformIsStackTransform( true );
                }
            }
        }
        
    }

}

#endif // end #ifndef _elxProbabilityAverageNormalizedMutualInformationMetric_HXX__
