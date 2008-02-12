#ifndef __elxGridSampler_hxx
#define __elxGridSampler_hxx

#include "elxGridSampler.h"

namespace elastix
{
  using namespace itk;


  /**
  * ******************* BeforeEachResolution ******************
  */

  template <class TElastix>
    void GridSampler<TElastix>
    ::BeforeEachResolution(void)
  {
    const unsigned int level =
      this->m_Registration->GetAsITKBaseType()->GetCurrentLevel();
    
    GridSpacingType gridspacing;

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
    this->SetSampleGridSpacing( gridspacing );

  } // end BeforeEachResolution


} // end namespace elastix

#endif // end #ifndef __elxGridSampler_hxx

