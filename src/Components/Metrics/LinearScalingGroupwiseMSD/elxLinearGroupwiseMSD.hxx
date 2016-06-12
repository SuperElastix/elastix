#ifndef __elxLinearGroupwiseMSD_HXX__
#define __elxLinearGroupwiseMSD_HXX__

#include "elxLinearGroupwiseMSD.h"

#include "itkHardLimiterFunction.h"
#include "itkExponentialLimiterFunction.h"

#include "itkTimeProbe.h"


namespace elastix
{
    /**
     * ****************** Constructor ***********************
     */
    
    template< class TElastix >
    LinearGroupwiseMSD< TElastix >
    ::LinearGroupwiseMSD()
    {
        
    }
    
    /**
     * ******************* Initialize ***********************
     */
    
    template< class TElastix >
    void
    LinearGroupwiseMSD< TElastix >
    ::Initialize( void ) throw ( itk::ExceptionObject )
    {
        itk::TimeProbe timer;
        timer.Start();
        this->Superclass1::Initialize();
        timer.Stop();
        elxout << "Initialization of LinearGroupwiseMSD metric took: " << static_cast< long >( timer.GetMean() * 1000 ) << " ms." << std::endl;
        
    }
        
    /**
     * ***************** BeforeRegistration ***********************
     */
        
    template< class TElastix >
    void
    LinearGroupwiseMSD< TElastix >
    ::BeforeRegistration( void )
    {        
        bool subtractMean = false;
        this->GetConfiguration()->ReadParameter( subtractMean, "SubtractMean", this->GetComponentLabel(), 0, 0 );
        this->SetSubtractMean( subtractMean );
        
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
    LinearGroupwiseMSD< TElastix >
    ::BeforeEachResolution( void )
    {
        unsigned int level = ( this->m_Registration->GetAsITKBaseType() )->GetCurrentLevel();

        std::string combination = "Sum";
        this->GetConfiguration()->ReadParameter( combination, "Combination", this->GetComponentLabel(), level, 0 );
        if( combination == "Sum" )
        {
            this->SetCombinationToSum();
        }
        else if( combination == "SquaredSum" )
        {
            this->SetCombinationToSqSum();
        }
        else if( combination == "SquareRootSum" )
        {
            this->SetCombinationToSqRSum();
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
        else if( templateImage == "Median" )
        {
            this->SetTemplateToMedian();
        }

        
        bool useRandomSampling = false;
        this->GetConfiguration()->ReadParameter( useRandomSampling, "SampleLastDimensionRandomly", this->GetComponentLabel(), level, 0 );
        this->SetSampleLastDimensionRandomly( useRandomSampling );
        
        int numSamplesLastDimension = 3;
        this->GetConfiguration()->ReadParameter( numSamplesLastDimension,"NumSamplesLastDimension", this->GetComponentLabel(), level, 0 );
        this->SetNumSamplesLastDimension( numSamplesLastDimension );
        
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

#endif // end #ifndef _elxLinearGroupwiseMSD_HXX__
