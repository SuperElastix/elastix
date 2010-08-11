/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/

#ifndef __elxReducedDimensionGridSampler_hxx
#define __elxReducedDimensionGridSampler_hxx

#include "elxReducedDimensionGridSampler.h"

#include "itkAdvancedBSplineDeformableTransform.h"

namespace elastix
{
  using namespace itk;

  /**
  * ******************* BeforeRegistration ******************
  */

  template <class TElastix>
  void ReducedDimensionGridSampler<TElastix>
    ::BeforeRegistration(void)
  {

    /** Get dimension to reduce from configuration. */
    unsigned int reducedDimension = InputImageDimension - 1;
    unsigned int reducedDimensionIndex = 0;
    this->GetConfiguration()->ReadParameter(
      reducedDimension, "ReducedDimension",
      this->GetComponentLabel(), 0, 0 );
    this->SetReducedDimension( reducedDimension );
    this->GetConfiguration()->ReadParameter(
      reducedDimensionIndex, "ReducedDimensionIndex",
      this->GetComponentLabel(), 0, 0 );
    this->SetReducedDimensionIndex( reducedDimensionIndex );

  }


  /**
  * ******************* BeforeEachResolution ******************
  */

  template <class TElastix>
    void ReducedDimensionGridSampler<TElastix>
    ::BeforeEachResolution(void)
  {
    const unsigned int level =
      this->m_Registration->GetAsITKBaseType()->GetCurrentLevel();

    GridSpacingType gridspacing;

    /** Read the automatic grid estimation settings. */
    bool automaticGridEstimation = false;
    this->GetConfiguration()->ReadParameter(
      automaticGridEstimation, "AutomaticSampleGridEstimation",
       this->GetComponentLabel(), 0, 0, false );
    float gridSpacingFactor = 1.0f;
    this->GetConfiguration()->ReadParameter(
      gridSpacingFactor, "GridSpacingFactor",
      this->GetComponentLabel(), level, 0, false );

    if ( automaticGridEstimation )
    {
      /** Set grid spacing to b-spline grid spacing. */
      typedef AdvancedBSplineDeformableTransform<typename ElastixType::CoordRepType, InputImageDimension, 2> BSpline2Type;
      typedef AdvancedBSplineDeformableTransform<typename ElastixType::CoordRepType, InputImageDimension, 3> BSpline3Type;

      typedef typename ElastixType::TransformBaseType TransformBaseType;
      typedef typename TransformBaseType::CombinationTransformType CombinationTransformType;

      CombinationTransformType * testPtr = this->GetElastix()->GetElxTransformBase()->GetAsCombinationTransform();
      if ( testPtr != NULL )
      {

        BSpline2Type * bspline2 =
          dynamic_cast< BSpline2Type * > ( testPtr->GetCurrentTransform() );
        BSpline3Type * bspline3 =
          dynamic_cast< BSpline3Type * > ( testPtr->GetCurrentTransform() );

        typename BSpline2Type::SpacingType spacing;
        if ( bspline2 != NULL )
        {
          spacing = bspline2->GetGridSpacing();
        }
        else if ( bspline3 != NULL )
        {
          spacing = bspline3->GetGridSpacing();
        }
        else
        {
          /** Not a bspline transform, fall back to reading grid spacing from parameter file. */
          automaticGridEstimation = false;
        }

        if ( automaticGridEstimation )
        {
          /** Compute the grid spacing in voxel units. */
          for ( unsigned int dim = 0; dim < InputImageDimension; ++dim )
          {
            if ( this->m_ReducedDimension == dim )
            {
              gridspacing[ dim ] = 1;
            }
            else
            {
              gridspacing[ dim ] = static_cast< int > ( gridSpacingFactor * spacing[ dim ] /
                this->GetElastix()->GetFixedImage()->GetSpacing()[ dim ] );
            }
         }
        }
        else
        {
          automaticGridEstimation = false;
        }
      }
    }

    if ( !automaticGridEstimation )
    {
      /** Read the desired grid spacing of the samples. */
      unsigned int spacing_dim;
      for ( unsigned int dim = 0; dim < InputImageDimension; dim++ )
      {
        spacing_dim = 2;
        this->GetConfiguration()->ReadParameter(
          spacing_dim, "SampleGridSpacing",
          this->GetComponentLabel(), level * InputImageDimension + dim, -1 );
        gridspacing[ dim ] = static_cast<SampleGridSpacingValueType>( spacing_dim );
      }
    }

    this->SetSampleGridSpacing( gridspacing );

    /** Output grid spacing to log file. */
    elxout << "Sample grid spacing of ReducedDimensionGridSampler = " << this->GetSampleGridSpacing() << std::endl;
  } // end BeforeEachResolution


} // end namespace elastix

#endif // end #ifndef __elxReducedDimensionGridSampler_hxx

