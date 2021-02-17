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

#ifndef elxResampleInterpolatorBase_h
#define elxResampleInterpolatorBase_h

/** Needed for the macros */
#include "elxMacro.h"

#include "elxBaseComponentSE.h"
#include "elxElastixBase.h"
#include "itkInterpolateImageFunction.h"

namespace elastix
{

/**
 * \class ResampleInterpolatorBase
 * \brief This class is the elastix base class for all ResampleInterpolators.
 *
 * This class contains all the common functionality for ResampleInterpolators.
 *
 * \ingroup ResampleInterpolators
 * \ingroup ComponentBaseClasses
 */

template <class TElastix>
class ITK_TEMPLATE_EXPORT ResampleInterpolatorBase : public BaseComponentSE<TElastix>
{
public:
  /** Standard ITK stuff. */
  typedef ResampleInterpolatorBase  Self;
  typedef BaseComponentSE<TElastix> Superclass;

  /** Run-time type information (and related methods). */
  itkTypeMacro(ResampleInterpolatorBase, BaseComponentSE);

  /** Typedef's from superclass. */
  typedef typename Superclass::ElastixType          ElastixType;
  typedef typename Superclass::ElastixPointer       ElastixPointer;
  typedef typename Superclass::ConfigurationType    ConfigurationType;
  typedef typename Superclass::ConfigurationPointer ConfigurationPointer;
  typedef typename Superclass::RegistrationType     RegistrationType;
  typedef typename Superclass::RegistrationPointer  RegistrationPointer;

  /** Typedef's from elastix. */
  typedef typename ElastixType::MovingImageType InputImageType;
  typedef ElastixBase::CoordRepType             CoordRepType;

  /** Other typedef's. */
  typedef itk::InterpolateImageFunction<InputImageType, CoordRepType> ITKBaseType;

  /** Typedef that is used in the elastix dll version. */
  typedef typename ElastixType::ParameterMapType ParameterMapType;

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


  /** Execute stuff before the actual transformation:
   * \li nothing here
   */
  virtual int
  BeforeAllTransformix(void)
  {
    return 0;
  }

  /** Function to read transform-parameters from a file. */
  virtual void
  ReadFromFile(void);

  /** Function to write transform-parameters to a file. */
  void
  WriteToFile(xl::xoutsimple & transformationParameterInfo) const;

  /** Function to create transform-parameters map. */
  void
  CreateTransformParametersMap(ParameterMapType & parameterMap) const;

protected:
  /** The constructor. */
  ResampleInterpolatorBase() = default;
  /** The destructor. */
  ~ResampleInterpolatorBase() override = default;

private:
  elxDeclarePureVirtualGetSelfMacro(ITKBaseType);

  virtual ParameterMapType
  CreateDerivedTransformParametersMap() const
  {
    return {};
  }

  /** The deleted copy constructor. */
  ResampleInterpolatorBase(const Self &) = delete;
  /** The deleted assignment operator. */
  void
  operator=(const Self &) = delete;
};

} // end namespace elastix

#ifndef ITK_MANUAL_INSTANTIATION
#  include "elxResampleInterpolatorBase.hxx"
#endif

#endif // end #ifndef elxResampleInterpolatorBase_h
