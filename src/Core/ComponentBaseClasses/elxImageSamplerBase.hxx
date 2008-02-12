#ifndef __elxImageSamplerBase_hxx
#define __elxImageSamplerBase_hxx

#include "elxImageSamplerBase.h"

namespace elastix
{
  //using namespace itk; not here because itk::ImageSamplerBase also exists. 

  /**
  * ******************* BeforeEachResolutionBase ******************
  */

  template <class TElastix>
    void ImageSamplerBase<TElastix>
    ::BeforeEachResolutionBase(void)
  {
    /** Get the current resolution level. */
    unsigned int level = 
      ( this->m_Registration->GetAsITKBaseType() )->GetCurrentLevel();

    /** Check if NewSamplesEveryIteration is possible with the selected ImageSampler. 
    * The "" argument means that no prefix is supplied. */
    bool newSamples = false;
    this->m_Configuration->ReadParameter( newSamples, "NewSamplesEveryIteration",
      "", level, 0, true );

    if ( newSamples )
    {
      bool ret = this->GetAsITKBaseType()->SelectingNewSamplesOnUpdateSupported();
      if ( !ret )
      {
        xl::xout["warning"]
        << "WARNING: You want to select new samples every iteration,\n"
          << "but the selected ImageSampler is not suited for that." 
          << std::endl;
      }
    }

  } // end BeforeEachResolutionBase

} // end namespace elastix

#endif //#ifndef __elxImageSamplerBase_hxx

