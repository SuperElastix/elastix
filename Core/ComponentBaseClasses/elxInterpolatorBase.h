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
  ITK_DISALLOW_COPY_AND_MOVE(InterpolatorBase);

  /** Standard ITK-stuff. */
  using Self = InterpolatorBase;
  using Superclass = BaseComponentSE<TElastix>;

  /** Run-time type information (and related methods). */
  itkTypeMacro(InterpolatorBase, BaseComponentSE);

  /** Typedefs inherited from Elastix. */
  using typename Superclass::ElastixType;
  using typename Superclass::RegistrationType;

  /** Other typedef's. */
  using InputImageType = typename ElastixType::MovingImageType;
  using CoordRepType = ElastixBase::CoordRepType;

  /** ITKBaseType. */
  using ITKBaseType = itk::InterpolateImageFunction<InputImageType, CoordRepType>;

  /** Retrieves this object as ITKBaseType. */
  ITKBaseType *
  GetAsITKBaseType()
  {
    return &(this->GetSelf());
  }


  /** Retrieves this object as ITKBaseType, to use in const functions. */
  const ITKBaseType *
  GetAsITKBaseType() const
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
};

} // end namespace elastix

#ifndef ITK_MANUAL_INSTANTIATION
#  include "elxInterpolatorBase.hxx"
#endif

#endif // end #ifndef elxInterpolatorBase_h
