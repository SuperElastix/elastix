#ifndef __itkGridScheduleComputer_TXX__
#define __itkGridScheduleComputer_TXX__


namespace itk
{

	/**
	 * ********************* Constructor ****************************
	 */
	
	template <unsigned int VImageDimension>
  GridScheduleComputer<VImageDimension>
  ::GridScheduleComputer()
	{
    this->m_BSplineOrder = 3;
    this->SetDefaultGridSpacingSchedule( 3, 16.0, 2.0 );

	} // end Constructor()
	

  /**
	 * ********************* SetDefaultGridSpacingSchedule ****************************
	 */
	
	template <unsigned int VImageDimension>
  void
  GridScheduleComputer<VImageDimension>
  ::SetDefaultGridSpacingSchedule(
    const unsigned int & levels,
    const SpacingType & finalGridSpacing,
    const float & upsamplingFactor )
	{
    /** Set member variables. */
    this->m_NumberOfLevels = levels;
    this->SetUpsamplingFactor( upsamplingFactor );
    this->m_GridSpacingScheduleIsDownwardsDivisible = true;

    /** Initialize the schedule. */
    this->m_GridSpacingSchedule.clear();
    this->m_GridSpacingSchedule.resize( levels, finalGridSpacing );

    /** Setup a default schedule. */
    float factor = this->m_UpsamplingFactor;
    for ( int i = levels - 2; i > -1; --i )
    {
      this->m_GridSpacingSchedule[ i ] *= factor;
      factor *= factor;
    }

    /** Determine if upsampling is required. */
    this->m_DoUpsampling.clear();
    if ( this->m_UpsamplingFactor > 1.0 )
    {
      this->m_DoUpsampling.resize( levels - 1, true );
    }
    else
    {
      this->m_DoUpsampling.resize( levels - 1, false );
    }

  } // end SetDefaultGridSpacingSchedule()

  
  /**
	 * ********************* SetGridSpacingSchedule ****************************
	 */
	
	template <unsigned int VImageDimension>
  void
  GridScheduleComputer<VImageDimension>
  ::SetGridSpacingSchedule( const VectorSpacingType & schedule )
	{
    /** Set member variables. */
    this->m_GridSpacingSchedule = schedule;
    this->m_NumberOfLevels = schedule.size();
    this->m_GridSpacingScheduleIsDownwardsDivisible = false;

    /** Determine if upsampling is required. */
    this->m_DoUpsampling.clear();
    this->m_DoUpsampling.resize( this->m_NumberOfLevels - 1, false );
    for ( unsigned int i = 0; i < this->m_NumberOfLevels - 1; ++i )
    {
      if ( schedule[ i ] != schedule[ i + 1 ] )
      {
        this->m_DoUpsampling[ i ] = true;
      }
    }

  } // end SetGridSpacingSchedule()


  /**
	 * ********************* GetGridSpacingSchedule ****************************
	 */
	
	template <unsigned int VImageDimension>
  void
  GridScheduleComputer<VImageDimension>
  ::GetGridSpacingSchedule( VectorSpacingType & schedule ) const
	{
    schedule = this->m_GridSpacingSchedule;

  } // end GetGridSpacingSchedule()


  /**
	 * ********************* ComputeBSplineGrid ****************************
	 */
	
	template <unsigned int VImageDimension>
  void
  GridScheduleComputer<VImageDimension>
  ::ComputeBSplineGrid( void )
	{
    /** Set the appropriate sizes. */
    this->m_GridOrigins.resize( this->m_NumberOfLevels );
    this->m_GridRegions.resize( this->m_NumberOfLevels );

    /** For all levels ... */
    for ( unsigned int i = 0; i < this->m_NumberOfLevels; ++i )
		{
      /** For all dimensions ... */
      SizeType size = this->m_Region.GetSize();
      SizeType gridsize;
      for ( unsigned int j = 0; j < Dimension; ++j )
      {
        /** Compute the grid size without the extra grid points at the edges. */
        const unsigned int bareGridSize = static_cast<unsigned int>(
          vcl_ceil( size[ j ] * this->m_Spacing[ j ] / this->m_GridSpacingSchedule[ i ][ j ] ) );

        /** The number of B-spline grid nodes is the bareGridSize plus the
         * B-spline order more grid nodes.
         */
        gridsize[ j ] = static_cast<SizeValueType>( bareGridSize + this->m_BSplineOrder );

        /** Compute the origin of the B-spline grid. */
        this->m_GridOrigins[ i ][ j ] = this->m_Origin[ j ] -
          this->m_GridSpacingSchedule[ i ][ j ]
          * vcl_floor( static_cast<double>( this->m_BSplineOrder ) / 2.0 );

        /** Shift the origin a little to the left, to place the grid
         * symmetrically on the image.
         */
        this->m_GridOrigins[ i ][ j ] -=
          ( this->m_GridSpacingSchedule[ i ][ j ] * bareGridSize
          - this->m_Spacing[ j ] * ( size[ j ] - 1 ) ) / 2.0;
        // todo: I don't get Stefans code at this point. Is it wrong?
      }

      /** Set the grid region. */
      this->m_GridRegions[ i ].SetSize( gridsize );
    }
    
  } // end ComputeBSplineGrid()
  
	/** Take into account the initial transform, if composition is used to 
		 * combine it with the current (bspline) transform *
		if ( (this->GetUseComposition()) && (this->Superclass1::GetInitialTransform() != 0) )
		{
			/** We have to determine a bounding box around the fixed image after
			 * applying the initial transform. This is done by iterating over the
			 * the boundary of the fixed image, evaluating the initial transform
			 * at those points, and keeping track of the minimum/maximum transformed
			 * coordinate in each dimension 
			 */

			/** Make a temporary copy; who knows, maybe some speedup can be achieved... *
			InitialTransformConstPointer initialTransform = 
				this->Superclass1::GetInitialTransform();
			/** The points that define the bounding box *
			InputPointType maxPoint;
			InputPointType minPoint;
			maxPoint.Fill( NumericTraits< CoordRepType >::NonpositiveMin() );
			minPoint.Fill( NumericTraits< CoordRepType >::max() );
			/** An iterator over the boundary of the fixed image *
			BoundaryIteratorType bit(fixedImage, fixedImageRegion);
			bit.SetExclusionRegionToInsetRegion();
			bit.GoToBegin();
			/** start loop over boundary; determines minPoint and maxPoint *
			while ( !bit.IsAtEnd() )
			{
				/** Get index, transform to physical point, apply initial transform
				 * NB: the OutputPointType of the initial transform by definition equals
				 * the InputPointType of this transform. *
        IndexType inputIndex = bit.GetIndex();
				InputPointType inputPoint;
				fixedImage->TransformIndexToPhysicalPoint(inputIndex, inputPoint);
				InputPointType outputPoint = 
					initialTransform->TransformPoint(	inputPoint );
				/** update minPoint and maxPoint *
				for ( unsigned int i = 0; i < SpaceDimension; i++ )
				{
					CoordRepType & outi = outputPoint[i];
					CoordRepType & maxi = maxPoint[i];
					CoordRepType & mini = minPoint[i];
					if ( outi > maxi )
					{
						maxi = outi;
					}
					if ( outi < mini )
					{
						mini = outi;
					}
				}
				/** step to next voxel *
				++bit;
			} //end while loop over fixed image boundary

			/** Set minPoint as new "fixedImageOrigin" (between quotes, since it
			 * is not really the origin of the fixedImage anymore) *
			fixedImageOrigin = minPoint;

			/** Compute the new "fixedImageSpacing" in each dimension *
			const double smallnumber = NumericTraits<double>::epsilon();
			for ( unsigned int i = 0; i < SpaceDimension; i++ )
			{
				/** Compute the length of the fixed image (in mm) for dimension i *
				double oldLength_i = 
					fixedImageSpacing[i] * static_cast<double>( fixedImageSize[i] - 1 );
				/** Compute the length of the bounding box (in mm) for dimension i *
        double newLength_i = static_cast<double>( maxPoint[i] - minPoint[i] );
				/** Scale the fixedImageSpacing by their ratio. *
				if (oldLength_i > smallnumber)
				{
					fixedImageSpacing[i] *= ( newLength_i / oldLength_i );
				}				
			}

		  /** We have now adapted the fixedImageOrigin and fixedImageSpacing.
			 * This makes sure that the BSpline grid is located at the position
			 * of the fixed image after undergoing the initial transform.  *
		     		  
		} // end if UseComposition && InitialTransform!=0



  /**
	 * ********************* GetBSplineGrid ****************************
	 */
	
	template <unsigned int VImageDimension>
  void
  GridScheduleComputer<VImageDimension>
  ::GetBSplineGrid(
    unsigned int level,
    RegionType & gridRegion,
    SpacingType & gridSpacing,
    OriginType & gridOrigin )
  {
    /** Check level. */
    if ( level > this->m_NumberOfLevels - 1 )
    {
      itkExceptionMacro(
        << "ERROR: Requesting resolution level "
        << level
        << ", but only "
        << this->m_NumberOfLevels
        << " levels exist." );
    }

    /** Return values. */
    gridRegion  = this->m_GridRegions[ level ];
    gridSpacing = this->m_GridSpacingSchedule[ level ];
    gridOrigin  = this->m_GridOrigins[ level ];

  } // end GetBSplineGrid()
  

  /**
	 * ********************* GetDoUpsampling ****************************
	 */
	
	template <unsigned int VImageDimension>
  bool
  GridScheduleComputer<VImageDimension>
  ::GetDoUpsampling( const unsigned int & level ) const
	{
    if ( level > this->m_NumberOfLevels - 1 )
    {
      return true;
    }
    return this->m_DoUpsampling[ level ];

  } // end GetDoUpsampling()


  /**
	 * ********************* PrintSelf ****************************
	 */
	
	template <unsigned int VImageDimension>
  void
  GridScheduleComputer<VImageDimension>
  ::PrintSelf( std::ostream & os, Indent indent ) const
	{
    Superclass::PrintSelf( os, indent );

    os << indent << "B-spline order: " << this->m_BSplineOrder << std::endl;
    os << indent << "NumberOfLevels: " << this->m_NumberOfLevels << std::endl;

    os << indent << "Spacing: " << this->m_Spacing << std::endl;
    os << indent << "Origin: " << this->m_Origin << std::endl;
    os << indent << "Region: " << std::endl;
    this->m_Region.Print( os, indent.GetNextIndent() );

    os << indent << "GridSpacingSchedule: " << std::endl;
    for ( unsigned int i = 0; i < this->m_NumberOfLevels; ++i )
    {
      os << indent.GetNextIndent() << this->m_GridSpacingSchedule[ i ] << std::endl;
    }

    os << indent << "GridOrigins: " << std::endl;
    for ( unsigned int i = 0; i < this->m_NumberOfLevels; ++i )
    {
      os << indent.GetNextIndent() << this->m_GridOrigins[ i ] << std::endl;
    }

    os << indent << "GridRegions: " << std::endl;
    for ( unsigned int i = 0; i < this->m_NumberOfLevels; ++i )
    {
      os << indent.GetNextIndent() << this->m_GridRegions[ i ] << std::endl;
    }

    os << indent << "UpsamplingFactor: " << this->m_UpsamplingFactor << std::endl;
    os << indent << "GridSpacingScheduleIsDownwardsDivisible: "
      << this->m_GridSpacingScheduleIsDownwardsDivisible << std::endl;

  } // end PrintSelf()

} // end namespace itk


#endif // end #ifndef __itkGridScheduleComputer_TXX__

