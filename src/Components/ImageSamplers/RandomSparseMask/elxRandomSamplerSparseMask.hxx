#ifndef __elxRandomSamplerSparseMask_hxx
#define __elxRandomSamplerSparseMask_hxx

#include "elxRandomSamplerSparseMask.h"

namespace elastix
{
  using namespace itk;


  /**
  * ******************* BeforeEachResolution ******************
  */

  template <class TElastix>
    void RandomSamplerSparseMask<TElastix>
    ::BeforeEachResolution(void)
  {
    const unsigned int level =
      ( this->m_Registration->GetAsITKBaseType() )->GetCurrentLevel();
  
    /** Set the NumberOfSpatialSamples. */
    unsigned long numberOfSpatialSamples = 5000;
    this->GetConfiguration()->ReadParameter( numberOfSpatialSamples,
      "NumberOfSpatialSamples", this->GetComponentLabel(), level, 0 );

    this->SetNumberOfSamples( numberOfSpatialSamples );

  } // end BeforeEachResolution


} // end namespace elastix

#endif // end #ifndef __elxRandomSamplerSparseMask_hxx

