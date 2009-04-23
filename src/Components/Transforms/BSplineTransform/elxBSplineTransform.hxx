/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even 
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/

#ifndef __elxBSplineTransform_hxx
#define __elxBSplineTransform_hxx

#include "elxBSplineTransform.h"

#include "itkImageRegionExclusionConstIteratorWithIndex.h"
#include "vnl/vnl_math.h"

namespace elastix
{
using namespace itk;


  /**
   * ********************* Constructor ****************************
   */
  
  template <class TElastix>
    BSplineTransform<TElastix>::
    BSplineTransform()
  {
    /** Initialize. */
    this->m_BSplineTransform = BSplineTransformType::New();
    this->SetCurrentTransform( this->m_BSplineTransform );

    this->m_GridScheduleComputer = GridScheduleComputerType::New();
    this->m_GridScheduleComputer->SetBSplineOrder( SplineOrder );

    this->m_GridUpsampler = GridUpsamplerType::New();
    this->m_GridUpsampler->SetBSplineOrder( SplineOrder );

  } // end Constructor()
  

  /**
   * ******************* BeforeRegistration ***********************
   */

  template <class TElastix>
    void BSplineTransform<TElastix>
    ::BeforeRegistration( void )
  {
    /** Set initial transform parameters to a 1x1x1 grid, with deformation (0,0,0).
     * In the method BeforeEachResolution() this will be replaced by the right grid size.
     * This seems not logical, but it is required, since the registration
     * class checks if the number of parameters in the transform is equal to
     * the number of parameters in the registration class. This check is done
     * before calling the BeforeEachResolution() methods.
     */
    
    /** Task 1 - Set the Grid. */

    /** Declarations. */
    RegionType gridregion;
    SizeType gridsize;
    IndexType gridindex;
    SpacingType gridspacing;
    OriginType gridorigin;
    
    /** Fill everything with default values. */
    gridsize.Fill(1);
    gridindex.Fill(0);
    gridspacing.Fill(1.0);
    gridorigin.Fill(0.0);
    
    /** Set it all. */
    gridregion.SetIndex( gridindex );
    gridregion.SetSize( gridsize );
    this->m_BSplineTransform->SetGridRegion( gridregion );
    this->m_BSplineTransform->SetGridSpacing( gridspacing );
    this->m_BSplineTransform->SetGridOrigin( gridorigin );
    
    /** Task 2 - Give the registration an initial parameter-array. */
    ParametersType dummyInitialParameters( this->GetNumberOfParameters() );
    dummyInitialParameters.Fill( 0.0 );
    
    /** Put parameters in the registration. */
    this->m_Registration->GetAsITKBaseType()
      ->SetInitialTransformParameters( dummyInitialParameters );

    /** Precompute the B-spline grid regions. */
    this->PreComputeGridInformation();
    
  } // end BeforeRegistration()
  

  /**
   * ***************** BeforeEachResolution ***********************
   */

  template <class TElastix>
    void BSplineTransform<TElastix>
    ::BeforeEachResolution( void )
  {
    /** What is the current resolution level? */
    unsigned int level = 
      this->m_Registration->GetAsITKBaseType()->GetCurrentLevel();

    /** Define the grid. */
    if ( level == 0 )
    {
      this->InitializeTransform();
    } 
    else
    {
      /** Upsample the B-spline grid, if required. */
      this->IncreaseScale();
    }

    /** Get the PassiveEdgeWidth and use it to set the OptimizerScales. */
    unsigned int passiveEdgeWidth = 0;
    this->GetConfiguration()->ReadParameter( passiveEdgeWidth,
      "PassiveEdgeWidth", this->GetComponentLabel(), level, 0 );
    this->SetOptimizerScales( passiveEdgeWidth );
  
  } // end BeforeEachResolution()
  

  /**
   * ******************** PreComputeGridInformation ***********************
   */

  template <class TElastix>
    void BSplineTransform<TElastix>::
    PreComputeGridInformation( void )
  {
    /** Get the total number of resolution levels. */
    unsigned int nrOfResolutions = 
      this->m_Registration->GetAsITKBaseType()->GetNumberOfLevels();

    /** Set up grid schedule computer with image info */
    this->m_GridScheduleComputer->SetImageOrigin(
      this->GetElastix()->GetFixedImage()->GetOrigin() );
    this->m_GridScheduleComputer->SetImageSpacing(
      this->GetElastix()->GetFixedImage()->GetSpacing() );
    this->m_GridScheduleComputer->SetImageRegion(
      this->GetElastix()->GetFixedImage()->GetLargestPossibleRegion() );

    /** Take the initial transform only into account, if composition is used. */
    if ( this->GetUseComposition() )
    {
      this->m_GridScheduleComputer->SetInitialTransform( this->Superclass1::GetInitialTransform() );
    }

    /** Get the grid spacing schedule from the parameter file.
     *
     * Method 1: The user specifies "FinalGridSpacingInVoxels"
     * Method 2: The user specifies "FinalGridSpacingInPhysicalUnits"
     * Method 3: The user specifies "FinalGridSpacing"
     *
     * Method 1 and 2 additionally take the "GridSpacingSchedule".
     * The GridSpacingSchedule is defined by downsampling factors 
     * for each resolution, for each dimension (just like the image
     * pyramid schedules). So, for 2D images, and 3 resulutions,
     * we can specify:
     * (GridSpacingSchedule 4.0 4.0 2.0 2.0 1.0 1.0)
     * Which is the default schedule, if no GridSpacingSchedule is supplied.
     *
     * Method 3 additionally takes the "UpsampleGridOption".
     * Method 3 is deprecated, but will be supported for a long while.
     * The "FinalGridSpacingInVoxels" has the same meaning as the former
     * "FinalGridSpacing".
     */

    /** determine method */
    bool method3 = false;
    unsigned int count = this->m_Configuration->
      CountNumberOfParameterEntries( "FinalGridSpacing" );
    if ( count != 0 )
    {
      method3 = true;
    }
    bool method2 = false;
    count = this->m_Configuration->
      CountNumberOfParameterEntries( "FinalGridSpacingInPhysicalUnits" );
    if ( count != 0 )
    {
      method2 = true;
    }
        
    /** Declare vars */
    SpacingType finalGridSpacingInVoxels;
    SpacingType finalGridSpacingInPhysicalUnits;
    finalGridSpacingInVoxels.Fill( 16.0 );    
    finalGridSpacingInPhysicalUnits.Fill( 8.0 ); // this default is never used

    if ( method2 )
    {
      /** Method 2:
       * Read the FinalGridSpacingInPhysicalUnits */
      for ( unsigned int dim = 0; dim < SpaceDimension; ++dim )
      {
        this->m_Configuration->ReadParameter(
          finalGridSpacingInPhysicalUnits[ dim ], "FinalGridSpacingInPhysicalUnits",
          this->GetComponentLabel(), dim , 0, false );
      }
    }
    else
    {

      /** Method 3:
       * Silently try to read the deprecated FinalGridSpacing (in voxels).
       */
      bool deprecatedused = false;
      for ( unsigned int dim = 0; dim < SpaceDimension; ++dim )
      {
        deprecatedused |= this->m_Configuration->ReadParameter(
          finalGridSpacingInVoxels[ dim ], "FinalGridSpacing",
          this->GetComponentLabel(), dim , 0, false );        
      }    
      if ( deprecatedused )
      {
        /** This message was introduced in elastix 3.9. Actual deprecation can only be done
         * after elastix 4.1 at least.
         */
        xl::xout["warning"] << "WARNING: deprecated parameter \"FinalGridSpacing\" used!\n"
          << "This parameter is scheduled to be removed in a future release!\n"
          << "Better use \"FinalGridSpacingInVoxels\", which has the same meaning, "
          << "or \"FinalGridSpacingInPhysicalUnits\"."
          << std::endl;
      }

      /** Method 1:
       * Read the FinalGridSpacingInVoxels
       */
      for ( unsigned int dim = 0; dim < SpaceDimension; ++dim )
      {
        this->m_Configuration->ReadParameter(
          finalGridSpacingInVoxels[ dim ], "FinalGridSpacingInVoxels",
          this->GetComponentLabel(), dim , 0 );
      }

      /** Compute grid spacing in physical units */
      for ( unsigned int dim = 0; dim < SpaceDimension; ++dim )
      {
        finalGridSpacingInPhysicalUnits[ dim ] = 
          finalGridSpacingInVoxels[ dim ] *
          this->GetElastix()->GetFixedImage()->GetSpacing()[ dim ];
      }

    } // method 1 or 3

    /** Set up a default grid spacing schedule. */
    this->m_GridScheduleComputer->SetDefaultSchedule(
      nrOfResolutions, 2.0 );
    GridScheduleType gridSchedule;
    this->m_GridScheduleComputer->GetSchedule( gridSchedule );
    
    /** Get the deprecated UpsampleGridOption. */
    this->ComputeGridSchedule_Deprecated( gridSchedule );
    
    /** Read what the user has specified. This overrules everything. */
    count = this->m_Configuration->
      CountNumberOfParameterEntries( "GridSpacingSchedule" );
    unsigned int entry_nr = 0;
    if ( count == 0 )
    {
      // keep the default schedule
    }
    else if ( count == nrOfResolutions )
    {
      for ( unsigned int res = 0; res < nrOfResolutions; ++res )
      {
        for ( unsigned int dim = 0; dim < SpaceDimension; ++dim )
        {
          this->m_Configuration->ReadParameter( gridSchedule[res][dim],
            "GridSpacingSchedule", entry_nr, false );       
        }
        ++entry_nr;
      }
    }
    else if ( count == nrOfResolutions*SpaceDimension )
    {
      for ( unsigned int res = 0; res < nrOfResolutions; ++res )
      {
        for ( unsigned int dim = 0; dim < SpaceDimension; ++dim )
        {
          this->m_Configuration->ReadParameter( gridSchedule[ res ][ dim ],
            "GridSpacingSchedule", entry_nr, false );
          ++entry_nr;
        }
      }
    }
    else
    {
      xl::xout["error"] 
        << "ERROR: Invalid GridSpacingSchedule! The number of entries"
        << " behind the GridSpacingSchedule option should equal the"
        << " numberOfResolutions, or the numberOfResolutions*imageDimension."
        << std::endl;
      itkExceptionMacro( << "ERROR: Invalid GridSpacingSchedule!" );
    }

    /** Set the grid schedule and final grid spacing in the schedule computer. */
    this->m_GridScheduleComputer->SetFinalGridSpacing( 
      finalGridSpacingInPhysicalUnits );
    this->m_GridScheduleComputer->SetSchedule( gridSchedule );

    /** Compute the necessary information. */
    this->m_GridScheduleComputer->ComputeBSplineGrid();
        
  } // end PreComputeGridInformation()

  
  /**
   * ******************** InitializeTransform ***********************
   *
   * Set the size of the initial control point grid and initialize
   * the parameters to 0.
   */

  template <class TElastix>
    void BSplineTransform<TElastix>::
    InitializeTransform( void )
  {
    /** Compute the B-spline grid region, origin, and spacing. */
    RegionType gridRegion;
    OriginType gridOrigin;
    SpacingType gridSpacing;
    this->m_GridScheduleComputer->GetBSplineGrid( 0,
      gridRegion, gridSpacing, gridOrigin );

    /** Set it in the BSplineTransform. */
    this->m_BSplineTransform->SetGridRegion( gridRegion );
    this->m_BSplineTransform->SetGridSpacing( gridSpacing );
    this->m_BSplineTransform->SetGridOrigin( gridOrigin );

    /** Set initial parameters for the first resolution to 0.0. */
    ParametersType initialParameters( this->GetNumberOfParameters() );
    initialParameters.Fill( 0.0 );
    this->m_Registration->GetAsITKBaseType()->
      SetInitialTransformParametersOfNextLevel( initialParameters );
    
  } // end InitializeTransform()
  
  
  /**
   * *********************** IncreaseScale ************************
   *
   * Upsample the grid of control points.
   */

  template <class TElastix>
    void BSplineTransform<TElastix>::
    IncreaseScale( void )
  {
    /** What is the current resolution level? */
    unsigned int level = 
      this->m_Registration->GetAsITKBaseType()->GetCurrentLevel();

    /** The current grid. */
    OriginType  currentGridOrigin  = this->m_BSplineTransform->GetGridOrigin();
    SpacingType currentGridSpacing = this->m_BSplineTransform->GetGridSpacing();
    RegionType  currentGridRegion  = this->m_BSplineTransform->GetGridRegion();

    /** The new required grid. */
    OriginType  requiredGridOrigin;
    SpacingType requiredGridSpacing;
    RegionType  requiredGridRegion;
    this->m_GridScheduleComputer->GetBSplineGrid( level,
      requiredGridRegion, requiredGridSpacing, requiredGridOrigin );

    /** Get the latest transform parameters. */
    ParametersType latestParameters =
      this->m_Registration->GetAsITKBaseType()->GetLastTransformParameters();

    /** Setup the GridUpsampler. */
    this->m_GridUpsampler->SetCurrentGridOrigin( currentGridOrigin );
    this->m_GridUpsampler->SetCurrentGridSpacing( currentGridSpacing );
    this->m_GridUpsampler->SetCurrentGridRegion( currentGridRegion );
    this->m_GridUpsampler->SetRequiredGridOrigin( requiredGridOrigin );
    this->m_GridUpsampler->SetRequiredGridSpacing( requiredGridSpacing );
    this->m_GridUpsampler->SetRequiredGridRegion( requiredGridRegion );

    /** Compute the upsampled B-spline parameters. */
    ParametersType upsampledParameters;
    this->m_GridUpsampler->UpsampleParameters( latestParameters, upsampledParameters );
    
    /** Set the new grid definition in the BSplineTransform. */
    this->m_BSplineTransform->SetGridOrigin( requiredGridOrigin );
    this->m_BSplineTransform->SetGridSpacing( requiredGridSpacing );
    this->m_BSplineTransform->SetGridRegion( requiredGridRegion );

    /** Set the initial parameters for the next level. */
    this->m_Registration->GetAsITKBaseType()->
      SetInitialTransformParametersOfNextLevel( upsampledParameters );

    /** Set the parameters in the BsplineTransform. */
    this->m_BSplineTransform->SetParameters(
      this->m_Registration->GetAsITKBaseType()->
      GetInitialTransformParametersOfNextLevel() );
  
  }  // end IncreaseScale()
  

  /**
   * ************************* ReadFromFile ************************
   */

  template <class TElastix>
  void BSplineTransform<TElastix>::
    ReadFromFile( void )
  {
    /** Read and Set the Grid: this is a BSplineTransform specific task. */

    /** Declarations. */
    RegionType  gridregion;
    SizeType    gridsize;
    IndexType   gridindex;
    SpacingType gridspacing;
    OriginType  gridorigin;
    
    /** Fill everything with default values. */
    gridsize.Fill( 1 );
    gridindex.Fill( 0 );
    gridspacing.Fill( 1.0 );
    gridorigin.Fill( 0.0 );

    /** Get GridSize, GridIndex, GridSpacing and GridOrigin. */
    for ( unsigned int i = 0; i < SpaceDimension; i++ )
    {
      this->m_Configuration->ReadParameter( gridsize[ i ], "GridSize", i );
      this->m_Configuration->ReadParameter( gridindex[ i ], "GridIndex", i );
      this->m_Configuration->ReadParameter( gridspacing[ i ], "GridSpacing", i );
      this->m_Configuration->ReadParameter( gridorigin[ i ], "GridOrigin", i );
    }
    
    /** Set it all. */
    gridregion.SetIndex( gridindex );
    gridregion.SetSize( gridsize );
    this->m_BSplineTransform->SetGridRegion( gridregion );
    this->m_BSplineTransform->SetGridSpacing( gridspacing );
    this->m_BSplineTransform->SetGridOrigin( gridorigin );

    /** Call the ReadFromFile from the TransformBase.
     * This must be done after setting the Grid, because later the
     * ReadFromFile from TransformBase calls SetParameters, which
     * checks the parameter-size, which is based on the GridSize.
     */
    this->Superclass2::ReadFromFile();

  } // end ReadFromFile()


  /**
   * ************************* WriteToFile ************************
   *
   * Saves the TransformParameters as a vector and if wanted
   * also as a deformation field.
   */

  template <class TElastix>
    void BSplineTransform<TElastix>::
    WriteToFile( const ParametersType & param ) const
  {
    /** Call the WriteToFile from the TransformBase. */
    this->Superclass2::WriteToFile( param );

    /** Add some BSplineTransform specific lines. */
    xout["transpar"] << std::endl << "// BSplineTransform specific" << std::endl;

    /** Get the GridSize, GridIndex, GridSpacing and
     * GridOrigin of this transform.
     */
    SizeType size = this->m_BSplineTransform->GetGridRegion().GetSize();
    IndexType index = this->m_BSplineTransform->GetGridRegion().GetIndex();
    SpacingType spacing = this->m_BSplineTransform->GetGridSpacing();
    OriginType origin = this->m_BSplineTransform->GetGridOrigin();

    /** Write the GridSize of this transform. */
    xout["transpar"] << "(GridSize ";
    for ( unsigned int i = 0; i < SpaceDimension - 1; i++ )
    {
      xout["transpar"] << size[ i ] << " ";
    }
    xout["transpar"] << size[ SpaceDimension - 1 ] << ")" << std::endl;
    
    /** Write the GridIndex of this transform. */
    xout["transpar"] << "(GridIndex ";
    for ( unsigned int i = 0; i < SpaceDimension - 1; i++ )
    {
      xout["transpar"] << index[ i ] << " ";
    }
    xout["transpar"] << index[ SpaceDimension - 1 ] << ")" << std::endl;
    
    /** Set the precision of cout to 2, because GridSpacing and
     * GridOrigin must have at least one digit precision.
     */
    xout["transpar"] << std::setprecision(10);

    /** Write the GridSpacing of this transform. */
    xout["transpar"] << "(GridSpacing ";
    for ( unsigned int i = 0; i < SpaceDimension - 1; i++ )
    {
      xout["transpar"] << spacing[ i ] << " ";
    }
    xout["transpar"] << spacing[ SpaceDimension - 1 ] << ")" << std::endl;

    /** Write the GridOrigin of this transform. */
    xout["transpar"] << "(GridOrigin ";
    for ( unsigned int i = 0; i < SpaceDimension - 1; i++ )
    {
      xout["transpar"] << origin[ i ] << " ";
    }
    xout["transpar"] << origin[ SpaceDimension - 1 ] << ")" << std::endl;

    /** Set the precision back to default value. */
    xout["transpar"] << std::setprecision(
      this->m_Elastix->GetDefaultOutputPrecision() );

  } // end WriteToFile()

  
  /**
   * ******************** ComputeGridSchedule_Deprecated *********************
   *
   * Computes schedule using UpsampleGridOption.
   */

  template <class TElastix>
    void
    BSplineTransform<TElastix>
    ::ComputeGridSchedule_Deprecated( GridScheduleType & schedule )
  {
    unsigned int count = this->m_Configuration->
      CountNumberOfParameterEntries( "UpsampleGridOption" );
    if ( count == 0 )
    {
      return;
    }
    
    /** If multigrid, then start with a lower resolution grid.
     * First, we have to find out what the resolution is for the initial grid,
     * i.e. the grid in the first resolution. This depends on the number of times
     * the grid has to be upsampled. The user can specify this amount with the
     * option "UpsampleGridOption".
     * - In case the user specifies only one such option
     * it is assumed that between all resolutions upsampling is needed or not.
     * This is also in line with former API (backwards compatability):
     *    (UpsampleGridOption "true") or (UpsampleGridOption "false")
     * In this case the factor is multiplied with 2^(nrOfResolutions - 1).
     * - In the other case the user has to specify after each resolution
     * whether or not the grid should be upsampled, i.e. (nrOfResolutions - 1)
     * times. For 4 resolutions this is done as:
     *    (UpsampleGridOption "true" "false" "true")
     * In this case the factor is multiplied with 2^(# of true's).
     * - In case nothing is given in the parameter-file, by default the
     * option (UpsampleGridOption "true") is assumed.
     */

    /** Get the number of resolutions. */
    int nrOfResolutions = static_cast<int>(
      this->GetRegistration()->GetAsITKBaseType()->GetNumberOfLevels() );
    unsigned int size = vnl_math_max( 1, nrOfResolutions - 1 );

    /** Fill upsampleBSplineGridOption. */
    bool tmp = true;
    bool deprecatedused = false;
    this->GetConfiguration()->ReadParameter( tmp, "UpsampleGridOption", 0 );
    std::vector< bool > upsampleGridOption( size, tmp );
    for ( unsigned int i = 1; static_cast<long>(i) < nrOfResolutions - 1; ++i )
    {
      tmp = upsampleGridOption[ i ];
      deprecatedused |= this->m_Configuration->ReadParameter(
        tmp, "UpsampleGridOption", i );
      upsampleGridOption[ i ] = tmp;
    }

    if ( deprecatedused )
    {
      /** This message was introduced in elastix 3.9. Actual deprecation can only be done
       * after elastix 4.1 at least.
       */
      xl::xout["warning"] << "WARNING: deprecated parameter \"UpsampleGridOption\" used!\n"
        << "This parameter is scheduled to be removed in a future release!\n"
        << "Better use \"GridSpacingSchedule\" instead."
        << std::endl;
    }

    /** Create a B-spline grid schedule using the upsample-grid-options. */
    schedule[ nrOfResolutions - 1 ].Fill( 1.0 );
    float factor = 1.0;
    for ( int res = nrOfResolutions - 2; res > -1; --res )
    {
      if ( upsampleGridOption[ res ] )
      {        
        factor *= 2.0;
      }      
      schedule[ res ].Fill( factor );
    }

  } // end ComputeGridSchedule_Deprecated()


  /**
   * *********************** SetOptimizerScales ***********************
   * Set the optimizer scales of the edge coefficients to infinity.
   */

  template <class TElastix>
    void BSplineTransform<TElastix>::
    SetOptimizerScales( unsigned int edgeWidth )
  {
    typedef ImageRegionExclusionConstIteratorWithIndex<ImageType>   IteratorType;
    typedef typename RegistrationType::ITKBaseType          ITKRegistrationType;
    typedef typename ITKRegistrationType::OptimizerType     OptimizerType;
    typedef typename OptimizerType::ScalesType              ScalesType;
    typedef typename ScalesType::ValueType                  ScalesValueType;

    /** Define new scales */
    const unsigned long numberOfParameters
      = this->m_BSplineTransform->GetNumberOfParameters();
    const unsigned long offset = numberOfParameters / SpaceDimension;
    ScalesType newScales( numberOfParameters );
    newScales.Fill( NumericTraits<ScalesValueType>::One );
    const ScalesValueType infScale = 10000.0;
    
    if ( edgeWidth == 0 )
    { 
      /** Just set the unit scales into the optimizer. */
      this->m_Registration->GetAsITKBaseType()->GetOptimizer()->SetScales( newScales );
      return;
    }

    /** Get the grid region information and create a fake coefficient image. */
    RegionType gridregion = this->m_BSplineTransform->GetGridRegion();
    SizeType gridsize = gridregion.GetSize();
    IndexType gridindex = gridregion.GetIndex();
    ImagePointer coeff = ImageType::New();
    coeff->SetRegions( gridregion );
    coeff->Allocate();
    
    /** Determine inset region. (so, the region with active parameters). */
    RegionType insetgridregion;
    SizeType insetgridsize;
    IndexType insetgridindex;
    for ( unsigned int i = 0; i < SpaceDimension; ++i )
    {
      insetgridsize[ i ] = static_cast<unsigned int>( vnl_math_max( 0, 
        static_cast<int>( gridsize[ i ] - 2 * edgeWidth ) ) );
      if ( insetgridsize[ i ] == 0 )
      {
        xl::xout["error"] 
          << "ERROR: you specified a PassiveEdgeWidth of "
          << edgeWidth
          << " while the total grid size in dimension " 
          << i
          << " is only "
          << gridsize[ i ] << "." << std::endl;
        itkExceptionMacro( << "ERROR: the PassiveEdgeWidth is too large!" );
      }
      insetgridindex[ i ] = gridindex[ i ] + edgeWidth;
    }
    insetgridregion.SetSize( insetgridsize );
    insetgridregion.SetIndex( insetgridindex );

    /** Set up iterator over the coefficient image. */
    IteratorType cIt( coeff, coeff->GetLargestPossibleRegion() );
    cIt.SetExclusionRegion( insetgridregion );
    cIt.GoToBegin();   
    
    /** Set the scales to infinity that correspond to edge coefficients
     * This (hopefully) makes sure they are not optimised during registration.
     */
    while ( !cIt.IsAtEnd() )
    {
      const IndexType & index = cIt.GetIndex();
      const unsigned long baseOffset = coeff->ComputeOffset( index );
      for ( unsigned int i = 0; i < SpaceDimension; ++i )
      {
        const unsigned int scalesIndex = static_cast<unsigned int>(
          baseOffset + i * offset );
        newScales[ scalesIndex ] = infScale;
      }
      ++cIt;
    }

    /** Set the scales into the optimizer. */
    this->m_Registration->GetAsITKBaseType()->GetOptimizer()->SetScales( newScales );

  } // end SetOptimizerScales()

  
} // end namespace elastix


#endif // end #ifndef __elxBSplineTransform_hxx

