#ifndef __elxMCNJE_HXX__
#define __elxMCNJE_HXX__

#include "elxMCNJE.h"

#include "itkTimeProbe.h"

namespace elastix
{
    /**
     * ****************** Constructor ***********************
     */
    
    template< class TElastix >
    MCNJE< TElastix >
    ::MCNJE()
    {
        
    }
    
    /**
     * ******************* Initialize ***********************
     */
    
    template< class TElastix >
    void
    MCNJE< TElastix >
    ::Initialize( void ) throw ( itk::ExceptionObject )
    {
        itk::TimeProbe timer;
        timer.Start();
        this->Superclass1::Initialize();
        timer.Stop();
        elxout << "Initialization of MCNJE metric took: "
        << static_cast< long >( timer.GetMean() * 1000 ) << " ms." << std::endl;
        
    }
        
    /**
     * ***************** BeforeRegistration ***********************
     */
        
    template< class TElastix >
    void
    MCNJE< TElastix >
    ::BeforeRegistration( void )
    {
        bool subtractMean = false;
        this->GetConfiguration()->ReadParameter( subtractMean, "SubtractMean", this->GetComponentLabel(), 0, 0 );
        this->SetSubtractMean( subtractMean );
        
        bool useFastAndLowMemoryVersion = false;
        this->GetConfiguration()->ReadParameter( useFastAndLowMemoryVersion, "UseFastAndLowMemoryVersion", this->GetComponentLabel(), 0, 0 );
        this->SetUseExplicitPDFDerivatives( !useFastAndLowMemoryVersion );
        
        int numberOfChannels = 3;
        this->GetConfiguration()->ReadParameter( numberOfChannels, "NumberOfChannels", this->GetComponentLabel(), 0, 0 );
        this->SetNumberOfChannels( numberOfChannels );

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
    MCNJE< TElastix >
    ::BeforeEachResolution( void )
    {
        unsigned int level = ( this->m_Registration->GetAsITKBaseType() )->GetCurrentLevel();

        std::string combination = "Average";
        this->GetConfiguration()->ReadParameter( combination, "Combination", this->GetComponentLabel(), level, 0 );
        if( combination == "Sum" )
        {
            this->SetCombinationToSum();
        }
        else if( combination == "Average" )
        {
            this->SetCombinationToAverage();
        }
        else if( combination == "SquaredSum" )
        {
            this->SetCombinationToSqSum();
        }
        else if( combination == "SquareRootSum" )
        {
            this->SetCombinationToSqRSum();
        }
        else
        {
            elxout << "Warning: Invalid combination defined and set to Average ";
            this->SetCombinationToAverage();
        }


        
        std::string templateImage = "ArithmeticAverage";
        this->GetConfiguration()->ReadParameter( templateImage, "TemplateImage", this->GetComponentLabel(), level, 0 );
        if( templateImage == "ArithmeticAverage" )
        {
            this->SetTemplateToAA();
        }
        else if( templateImage == "GeometricAverage" )
        {
            this->SetTemplateToGA();
        }
        else if( templateImage == "HarmonicAverage" )
        {
            this->SetTemplateToHA();
        }
        else if( templateImage == "ReducedArithmeticAverage" )
        {
            this->SetTemplateToReducedAA();
        }
        else if( templateImage == "Median" )
        {
            elxout << "Warning: Experimental feature ";
            this->SetTemplateToMedian();
        }
        else
        {
            elxout << "Warning: Invalid template defined and set to arithmetic average ";
            this->SetTemplateToAA();
        }

        
        bool useRandomSampling = false;
        this->GetConfiguration()->ReadParameter( useRandomSampling, "SampleLastDimensionRandomly", this->GetComponentLabel(), level, 0 );
        this->SetSampleLastDimensionRandomly( useRandomSampling );
        
        int numSamplesLastDimension = 3;
        this->GetConfiguration()->ReadParameter( numSamplesLastDimension,"NumSamplesLastDimension", this->GetComponentLabel(), level, 0 );
        this->SetNumSamplesLastDimension( numSamplesLastDimension );
        
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
        
        CombinationTransformType * testPtr1
        = dynamic_cast< CombinationTransformType * >( this->GetElastix()->GetElxTransformBase() );
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
                    
                    if( testPtr3->GetNumberOfSubTransforms() > 0 )
                    {
                        ReducedDimensionBSplineTransformBaseType * testPtr4 = dynamic_cast< ReducedDimensionBSplineTransformBaseType * >(testPtr3->GetSubTransform( 0 ).GetPointer() );
                        if( testPtr4 )
                        {
                            FixedImageSizeType gridSize;
                            gridSize.Fill( testPtr3->GetNumberOfSubTransforms() );
                            this->SetGridSize( gridSize );
                        }
                    }
                }
            }
        }
    }
}

#endif // end #ifndef _elxMCNJE_HXX__
