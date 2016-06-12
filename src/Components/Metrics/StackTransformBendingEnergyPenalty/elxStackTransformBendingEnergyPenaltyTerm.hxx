#ifndef __elxStackTransformBendingEnergyPenaltyTerm_HXX__
#define __elxStackTransformBendingEnergyPenaltyTerm_HXX__

#include "elxStackTransformBendingEnergyPenaltyTerm.h"

#include "itkTimeProbe.h"

namespace elastix
{
    
    /**
     * ****************** Constructor ***********************
     */
    
    template< class TElastix >
    StackTransformBendingEnergyPenalty< TElastix >
    ::StackTransformBendingEnergyPenalty()
    {
        
    }


/**
 * ******************* Initialize ***********************
 */

template< class TElastix >
void
StackTransformBendingEnergyPenalty< TElastix >
::Initialize( void ) throw ( itk::ExceptionObject )
{
  itk::TimeProbe timer;
  timer.Start();
  this->Superclass1::Initialize();
  timer.Stop();
  elxout << "Initialization of TransformBendingEnergy metric took: "
    << static_cast< long >( timer.GetMean() * 1000 ) << " ms." << std::endl;

} // end Initialize()

    /**
     * ***************** BeforeRegistration ***********************
     */
    
    template< class TElastix >
    void
    StackTransformBendingEnergyPenalty< TElastix >
    ::BeforeRegistration( void )
    {
        
        bool subtractMean = false;
        this->GetConfiguration()->ReadParameter( subtractMean, "SubtractMean", this->GetComponentLabel(), 0, 0 );
        this->SetSubtractMean( subtractMean );
                
    }
    
/**
 * ***************** BeforeEachResolution ***********************
 */

template< class TElastix >
void
StackTransformBendingEnergyPenalty< TElastix >
::BeforeEachResolution( void )
{
  /** Get the current resolution level. */
  unsigned int level
    = ( this->m_Registration->GetAsITKBaseType() )->GetCurrentLevel();
    
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
            this->SetTransformIsBSpline( true );
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
                        this->SetSubTransformIsBSpline( true );

                    }
                }
            }
        }
    }


} // end BeforeEachResolution()


} // end namespace elastix

#endif // end #ifndef __elxStackTransformBendingEnergyPenaltyTerm_HXX__
