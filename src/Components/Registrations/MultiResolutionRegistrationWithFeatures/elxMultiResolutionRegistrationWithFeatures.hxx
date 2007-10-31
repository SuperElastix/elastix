#ifndef __elxMultiResolutionRegistrationWithFeatures_HXX__
#define __elxMultiResolutionRegistrationWithFeatures_HXX__

#include "elxMultiResolutionRegistrationWithFeatures.h"

namespace elastix
{
using namespace itk;
	
	/**
	 * ******************* BeforeRegistration ***********************
	 */

	template <class TElastix>
		void MultiResolutionRegistrationWithFeatures<TElastix>
		::BeforeRegistration( void )
	{	
		/** Get the components from this->m_Elastix and set them. */
		this->GetAndSetComponents();

		/** Set the number of resolutions. */
		unsigned int numberOfResolutions = 3;
		this->m_Configuration->ReadParameter( numberOfResolutions, "NumberOfResolutions", 0 );
		this->SetNumberOfLevels( numberOfResolutions );
				
		/** Set the FixedImageRegions to the buffered regions. */
    this->GetAndSetFixedImageRegions();
	
    /** Set the fixed image interpolators. */
    this->GetAndSetFixedImageInterpolators();
	
	} // end BeforeRegistration()


  /**
	 * ******************* BeforeEachResolution ***********************
	 */

	template <class TElastix>
		void MultiResolutionRegistrationWithFeatures<TElastix>
		::BeforeEachResolution( void )
	{	
    /** Get the current resolution level. */
		unsigned int level = this->GetCurrentLevel();

    /** Set the masks in the metric. */
    this->UpdateFixedMasks( level );
    this->UpdateMovingMasks( level );

  } // end BeforeEachResolution()
	
	
	/**
	 * *********************** GetAndSetComponents ************************
	 */

	template <class TElastix>
		void MultiResolutionRegistrationWithFeatures<TElastix>
		::GetAndSetComponents( void )
	{
    /** Get the component from this->GetElastix() (as elx::...BaseType *),
     * cast it to the appropriate type and set it in 'this'.
     */

    /** Set the metric. */
    this->SetMetric( this->GetElastix()->
      GetElxMetricBase()->GetAsITKBaseType() );

    /** Set the fixed images. */
    for ( unsigned int i = 0; i < this->GetElastix()->GetNumberOfFixedImages(); ++i )
    {
      this->SetFixedImage( this->GetElastix()->GetFixedImage( i ), i );
    }

    /** Set the moving images. */
    for ( unsigned int i = 0; i < this->GetElastix()->GetNumberOfMovingImages(); ++i )
    {
      this->SetMovingImage( this->GetElastix()->GetMovingImage( i ), i );
    }

    /** Set the fixed image pyramids. */
    for ( unsigned int i = 0; i < this->GetElastix()->GetNumberOfFixedImagePyramids(); ++i )
    {
      this->SetFixedImagePyramid( this->GetElastix()->
        GetElxFixedImagePyramidBase( i )->GetAsITKBaseType(), i );
    }

    /** Set the moving image pyramids. */
    for ( unsigned int i = 0; i < this->GetElastix()->GetNumberOfMovingImagePyramids(); ++i )
    {
      this->SetMovingImagePyramid( this->GetElastix()->
        GetElxMovingImagePyramidBase( i )->GetAsITKBaseType(), i );
    }
     
    /** Set the moving image interpolators. */
    for ( unsigned int i = 0; i < this->GetElastix()->GetNumberOfInterpolators(); ++i )
    {
      this->SetInterpolator( this->GetElastix()->
        GetElxInterpolatorBase( i )->GetAsITKBaseType(), i );
    }

    /** Set the optimizer. */
    this->SetOptimizer( dynamic_cast<OptimizerType*>(
      this->GetElastix()->GetElxOptimizerBase()->GetAsITKBaseType() ) );
    
    /** Set the transform. */
    this->SetTransform( this->GetElastix()->
      GetElxTransformBase()->GetAsITKBaseType() );
    
	} // end GetAndSetComponents()


  /**
	 * *********************** GetAndSetFixedImageRegions ************************
	 */

	template <class TElastix>
		void MultiResolutionRegistrationWithFeatures<TElastix>
		::GetAndSetFixedImageRegions( void )
	{
    for ( unsigned int i = 0; i < this->GetElastix()->GetNumberOfFixedImages(); ++i )
    {
      /** Make sure the fixed image is up to date. */
		  try
		  {
			  this->GetElastix()->GetFixedImage( i )->Update();
		  }
		  catch( itk::ExceptionObject & excp )
		  {
			  /** Add information to the exception. */
        excp.SetLocation( "MultiResolutionRegistrationWithFeatures - BeforeRegistration()" );
			  std::string err_str = excp.GetDescription();
			  err_str += "\nError occured while updating region info of the fixed image.\n";
			  excp.SetDescription( err_str );
			  /** Pass the exception to an higher level. */
			  throw excp;
		  }

		  /** Set the fixed image region. */
		  this->SetFixedImageRegion( this->GetElastix()->GetFixedImage( i )->GetBufferedRegion(), i );
    }

  } // end GetAndSetFixedImageRegions()


  /**
	 * *********************** GetAndSetFixedImageInterpolators ************************
	 */

	template <class TElastix>
		void MultiResolutionRegistrationWithFeatures<TElastix>
		::GetAndSetFixedImageInterpolators( void )
	{
    /** Shrot cut. */
    const unsigned int noFixIm = this->GetNumberOfFixedImages();

    /** Get the spline order of the fixed feature image interpolators. */
    unsigned int splineOrder = 1;
    this->m_Configuration->ReadParameter(
      splineOrder, "FixedImageInterpolatorBSplineOrder", 0 );
    std::vector< unsigned int > soFII( noFixIm, splineOrder );
    for ( unsigned int i = 1; i < noFixIm; ++i )
    {
      this->m_Configuration->ReadParameter(
        soFII[ i ], "FixedImageInterpolatorBSplineOrder", i, true );
    }

    /** Create and set interpolators for the fixed feature images. */
    typedef BSplineInterpolateImageFunction< FixedImageType >             FixedImageInterpolatorType;
    typedef std::vector< typename FixedImageInterpolatorType::Pointer >   FixedImageInterpolatorVectorType;
    FixedImageInterpolatorVectorType interpolators( noFixIm );
    for ( unsigned int i = 0; i < noFixIm; i++ )
    {
      interpolators[ i ] = FixedImageInterpolatorType::New();
      interpolators[ i ]->SetSplineOrder( soFII[ i ] );
      this->SetFixedImageInterpolator( interpolators[ i ], i );
    }

  } // end GetAndSetFixedImageInterpolators()


  /**
	 * ************************* UpdateFixedMasks ************************
   */

  template <class TElastix>
		void MultiResolutionRegistrationWithFeatures<TElastix>
    ::UpdateFixedMasks( unsigned int level )
	{    
    /** some shortcuts. */
    const unsigned int nrOfFixedMasks = this->GetElastix()->GetNumberOfFixedMasks();
    const unsigned int nrOfFixedImages = this->GetElastix()->GetNumberOfFixedImages();
    const unsigned int nrOfFixedImagePyramids = this->GetElastix()->GetNumberOfFixedImagePyramids();

    /** Array of bools, that remembers for each mask if erosion is wanted. */
    UseMaskErosionArrayType useMaskErosionArray;

    /** Bool that remembers if mask erosion is wanted in any of the masks 
     * remains false when no masks are used.
     */
    bool useMaskErosion;
    
    /** Read whether mask erosion is wanted, if any masks were supplied. */
    useMaskErosion = this->ReadMaskParameters( useMaskErosionArray,
      nrOfFixedMasks, "Fixed", level );

    /** Create and start timer, to time the whole mask configuration procedure. */
    TimerPointer timer = TimerType::New();
    timer->StartTimer();
 
    /** Now set the masks. *
    if (  ( nrOfFixedImages == 1 || nrOfFixedMasks == 0 ) &&
      nrOfFixedMasks <= 1 &&
      ( nrOfFixedImagePyramids == 1  || !useMaskErosion || nrOfFixedMasks == 0 ) )
    {
      /** 1 image || nomask, <= 1 mask, 1 pyramid || noerosion || nomask: 
       * --> we can use one mask for all metrics! (or no mask at all).
       *
      FixedMaskSpatialObjectPointer fixedMask = this->GenerateFixedMaskSpatialObject( 
        this->GetElastix()->GetFixedMask(), useMaskErosion,
        this->GetFixedImagePyramid(), level );
      this->GetCombinationMetric()->SetFixedImageMask( fixedMask );
    }
    else if ( nrOfFixedImages == 1 && nrOfFixedMasks == 1 )
    {
      /** 1 image, 1 mask, erosion && multiple pyramids
       * Set a differently eroded mask in each metric. The eroded
       * masks are all based on the same mask image, but generated with
       * different pyramid settings.
       *
      for ( unsigned int i = 0; i < nrOfMetrics; ++i )
      { 
        FixedMaskSpatialObjectPointer fixedMask = this->GenerateFixedMaskSpatialObject( 
          this->GetElastix()->GetFixedMask(), useMaskErosion,
          this->GetFixedImagePyramid( i ), level );
        this->GetCombinationMetric()->SetFixedImageMask( fixedMask, i );
      }
    }
    else
    {
      /** All other cases. Note that the number of pyramids should equal 1 or
       * should equal the number of metrics. 
       * Set each supplied mask in its corresponding metric, possibly after erosion.
       * If more metrics than masks are present, the last metrics will not use a mask.
       * If less metrics than masks are present, the last masks will be ignored.
       *
      for ( unsigned int i = 0; i < nrOfMetrics; ++i)
      {
        bool useMask_i = false; // default value in case of more metrics than masks
        if ( i < nrOfFixedMasks )
        {
          useMask_i = useMaskErosionArray[ i ];
        }
        FixedImagePyramidPointer pyramid_i = this->GetFixedImagePyramid(); // default value in case of only 1 pyramid
        if ( i < nrOfFixedImagePyramids )
        {
          pyramid_i = this->GetFixedImagePyramid( i );
        }
        FixedMaskSpatialObjectPointer fixedMask = this->GenerateFixedMaskSpatialObject( 
          this->GetElastix()->GetFixedMask( i ), useMask_i, pyramid_i, level );
        this->GetCombinationMetric()->SetFixedImageMask( fixedMask, i );
      }
    } // end else

    /** Stop timer and print the elapsed time. */
		timer->StopTimer();
    elxout << "Setting the fixed masks took: "
		  << static_cast<long>( timer->GetElapsedClockSec() * 1000 ) 
      << " ms." << std::endl;
    	
  } // end UpdateFixedMasks()
  

 	/**
	 * ************************* UpdateMovingMasks ************************
   */

  template <class TElastix>
		void MultiResolutionRegistrationWithFeatures<TElastix>
    ::UpdateMovingMasks( unsigned int level )
	{    
    /** some shortcuts */
    const unsigned int nrOfMetrics = this->GetElastix()->GetNumberOfMetrics();
    const unsigned int nrOfMovingMasks = this->GetElastix()->GetNumberOfMovingMasks();
    const unsigned int nrOfMovingImages = this->GetElastix()->GetNumberOfMovingImages();
    const unsigned int nrOfMovingImagePyramids = this->GetElastix()->GetNumberOfMovingImagePyramids();

    /** Array of bools, that remembers for each mask if erosion is wanted */
    UseMaskErosionArrayType useMaskErosionArray;

    /** Bool that remembers if mask erosion is wanted in any of the masks 
     * remains false when no masks are used */
    bool useMaskErosion;
    
    /** Read whether mask erosion is wanted, if any masks were supplied */
    useMaskErosion = this->ReadMaskParameters( useMaskErosionArray,
      nrOfMovingMasks, "Moving", level);
    
    /** Create and start timer, to time the whole mask configuration procedure. */
    TimerPointer timer = TimerType::New();
    timer->StartTimer();
        
    /** Now set the masks *
    if (  ( (nrOfMovingImages==1) || (nrOfMovingMasks==0) ) &&
          (nrOfMovingMasks<=1) &&
          ( (nrOfMovingImagePyramids==1) || !useMaskErosion || (nrOfMovingMasks==0) )   )
    {      
      /** 1 image || nomask, <=1 mask, 1 pyramid || noerosion || nomask: 
       * --> we can use one mask for all metrics! (or no mask at all) *
      MovingMaskSpatialObjectPointer movingMask = this->GenerateMovingMaskSpatialObject( 
        this->GetElastix()->GetMovingMask(), useMaskErosion,
        this->GetMovingImagePyramid(), level );
      this->GetCombinationMetric()->SetMovingImageMask( movingMask );
    } 
    else if ( (nrOfMovingImages==1) && (nrOfMovingMasks==1) )
    {
      /** 1 image, 1 mask, erosion && multiple pyramids
       * Set a differently eroded mask in each metric. The eroded
       * masks are all based on the same mask image, but generated with
       * different pyramid settings. *
      for (unsigned int i = 0; i < nrOfMetrics; ++i)
      { 
        MovingMaskSpatialObjectPointer movingMask = this->GenerateMovingMaskSpatialObject( 
          this->GetElastix()->GetMovingMask(), useMaskErosion,
          this->GetMovingImagePyramid(i), level );
        this->GetCombinationMetric()->SetMovingImageMask( movingMask, i );
      }
    }
    else
    {
      /** All other cases. Note that the number of pyramids should equal 1 or
       * should equal the number of metrics. 
       * Set each supplied mask in its corresponding metric, possibly after erosion.
       * If more metrics than masks are present, the last metrics will not use a mask.
       * If less metrics than masks are present, the last masks will be ignored. *
      for (unsigned int i = 0; i < nrOfMetrics; ++i)
      {
        bool useMask_i = false; // default value in case of more metrics than masks
        if ( i < nrOfMovingMasks )
        {
          useMask_i = useMaskErosionArray[i];
        }
        MovingImagePyramidPointer pyramid_i = this->GetMovingImagePyramid(); // default value in case of only 1 pyramid
        if (i < nrOfMovingImagePyramids)
        {
          pyramid_i = this->GetMovingImagePyramid(i);
        }
        MovingMaskSpatialObjectPointer movingMask = this->GenerateMovingMaskSpatialObject( 
          this->GetElastix()->GetMovingMask(i), useMask_i, pyramid_i, level );
        this->GetCombinationMetric()->SetMovingImageMask( movingMask, i );
      }
    } // end else

    /** Stop timer and print the elapsed time. */
		timer->StopTimer();
    elxout << "Setting the moving masks took: "
		  << static_cast<long>( timer->GetElapsedClockSec() * 1000 ) 
      << " ms." << std::endl;
    	
  } // end UpdateMovingMasks()


} // end namespace elastix

#endif // end #ifndef __elxMultiResolutionRegistrationWithFeatures_HXX__

