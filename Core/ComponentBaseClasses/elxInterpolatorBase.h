/*=========================================================================
 *
 *  Copyright UMC Utrecht and contributors
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *        http://www.apache.org/licenses/LICENSE-2.0.txt
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *=========================================================================*/

#ifndef elxInterpolatorBase_h
#define elxInterpolatorBase_h

/** Needed for the macros */
#include "elxMacro.h"

#include "elxBaseComponentSE.h"
#include "elxElastixBase.h"

#include "itkInterpolateImageFunction.h"

namespace elastix
{

/**
 * \class InterpolatorBase
 * \brief This class is the elastix base class for all Interpolators.
 *
 * This class contains all the common functionality for Interpolators.
 *
 * \ingroup Interpolators
 * \ingroup ComponentBaseClasses
 */

template <class TElastix>
class ITK_TEMPLATE_EXPORT InterpolatorBase : public BaseComponentSE<TElastix>
{
public:
  /** Standard ITK-stuff. */
  typedef InterpolatorBase          Self;
  typedef BaseComponentSE<TElastix> Superclass;

  /** Run-time type information (and related methods). */
  itkTypeMacro(InterpolatorBase, BaseComponentSE);

  /** Typedefs inherited from Elastix. */
  typedef typename Superclass::ElastixType          ElastixType;
  typedef typename Superclass::ElastixPointer       ElastixPointer;
  typedef typename Superclass::ConfigurationType    ConfigurationType;
  typedef typename Superclass::ConfigurationPointer ConfigurationPointer;
  typedef typename Superclass::RegistrationType     RegistrationType;
  typedef typename Superclass::RegistrationPointer  RegistrationPointer;

  /** Other typedef's. */
  typedef typename ElastixType::MovingImageType InputImageType;
  typedef ElastixBase::CoordRepType             CoordRepType;

  /** ITKBaseType. */
  typedef itk::InterpolateImageFunction<InputImageType, CoordRepType> ITKBaseType;

  /** Retrieves this object as ITKBaseType. */
  ITKBaseType *
  GetAsITKBaseType(void)
  {
    return &(this->GetSelf());
  }


  /** Retrieves this object as ITKBaseType, to use in const functions. */
  const ITKBaseType *
  GetAsITKBaseType(void) const
  {
    return &(this->GetSelf());
  }


protected:
  /** The constructor. */
  InterpolatorBase() = default;
  /** The destructor. */
  ~InterpolatorBase() override = default;

private:
  elxDeclarePureVirtualGetSelfMacro(ITKBaseType);

  /** The deleted copy constructor. */
  InterpolatorBase(const Self &) = delete;
  /** The deleted assignment operator. */
  void
  operator=(const Self &) = delete;
};

} // end namespace elastix

#ifndef ITK_MANUAL_INSTANTIATION
#  include "elxInterpolatorBase.hxx"
#endif

#endif // end #ifndef elxInterpolatorBase_h
