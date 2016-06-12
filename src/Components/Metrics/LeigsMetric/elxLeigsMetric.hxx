#ifndef __elxLeigsMetric_HXX__
#define __elxLeigsMetric_HXX__

#include "elxLeigsMetric.h"

#include "itkTimeProbe.h"

namespace elastix
{
    /**
     * ****************** Constructor ***********************
     */
    
    template< class TElastix >
    LeigsMetric< TElastix >
    ::LeigsMetric()
    {
        
    }
    
    /**
     * ******************* Initialize ***********************
     */
    
    template< class TElastix >
    void
    LeigsMetric< TElastix >
    ::Initialize( void ) throw ( itk::ExceptionObject )
    {
        itk::TimeProbe timer;
        timer.Start();
        this->Superclass1::Initialize();
        timer.Stop();
        elxout << "Initialization of LeigsMetric metric took: "
        << static_cast< long >( timer.GetMean() * 1000 ) << " ms." << std::endl;
        
    }
        
    /**
     * ***************** BeforeRegistration ***********************
     */
        
    template< class TElastix >
    void
    LeigsMetric< TElastix >
    ::BeforeRegistration( void )
    {
        bool subtractMean = false;
        this->GetConfiguration()->ReadParameter( subtractMean, "SubtractMean", this->GetComponentLabel(), 0, 0 );
        this->SetSubtractMean( subtractMean );
        
        /** Get the tree type. */
        std::string treeType = "KDTree";
        this->m_Configuration->ReadParameter( treeType, "TreeType", 0 );
        if( treeType == "KDTree" )
        {
            this->SetANNkDTree();
        }
        if( treeType == "BruteForce" )
        {
            this->SetANNBruteForceTree();
        }
        
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
    LeigsMetric< TElastix >
    ::BeforeEachResolution( void )
    {
        unsigned int level = ( this->m_Registration->GetAsITKBaseType() )->GetCurrentLevel();
        
        bool useRandomSampling = false;
        this->GetConfiguration()->ReadParameter( useRandomSampling, "SampleLastDimensionRandomly", this->GetComponentLabel(), level, 0 );
        this->SetSampleLastDimensionRandomly( useRandomSampling );

        int numSamplesLastDimension = 3;
        this->GetConfiguration()->ReadParameter( numSamplesLastDimension,"NumSamplesLastDimension", this->GetComponentLabel(), level, 0 );
        this->SetNumSamplesLastDimension( numSamplesLastDimension );

        int nearestNeighbours = 2;
        this->GetConfiguration()->ReadParameter( nearestNeighbours,"NearestNeighbours", this->GetComponentLabel(), level, 0 );
        this->SetNearestNeighbours( nearestNeighbours );

        double time = 1;
        this->GetConfiguration()->ReadParameter( time,"Time", this->GetComponentLabel(), level, 0 );
        this->SetTime( time );

        double treeError = 0.0001;
        this->GetConfiguration()->ReadParameter( treeError,"TreeError", this->GetComponentLabel(), level, 0 );
        std::string treeSearchType = "Standard";
        this->m_Configuration->ReadParameter( treeSearchType, "TreeSearchType", level, 0 );
        if( treeSearchType == "Standard" )
        {
            this->SetANNStandardTreeSearch( nearestNeighbours, treeError );
        }

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

#endif // end #ifndef _elxLeigsMetric_HXX__
