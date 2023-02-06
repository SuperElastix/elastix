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

#include "elxComponentLoader.h"
#include "elxSupportedImageTypes.h"
#include "elxInstallFunctions.h"
#include "elxMacro.h"
#include "elxInstallAllComponents.h"

// ITKFactoryRegistration headers:
#include <itkImageIOFactoryRegisterManager.h>
#include <itkMeshIOFactoryRegisterManager.h>
#include <itkTransformIOFactoryRegisterManager.h>

#include <iostream>
#include <string>
#include <tuple>

namespace elastix
{
/**
 * Definition of class template, needed in InstallSupportedImageTypes()
 */

/** Define a class<N> with a method DO(...) that calls class<N+1>::DO(...) */
template <ComponentDatabase::IndexType VIndex>
class _installsupportedimagesrecursively
{
public:
  /** ElastixTypedef is defined in elxSupportedImageTypes.h, by means of the
   * the elxSupportedImageTypesMacro */
  using ET = ElastixTypedef<VIndex>;
  using ElastixType = typename ET::ElastixType;
  using ComponentDescriptionType = ComponentDatabase::ComponentDescriptionType;

  static int
  DO(const ComponentDescriptionType & name, ComponentDatabase * cdb)
  {
    int dummy1 = InstallFunctions<ElastixType>::InstallComponent(name, VIndex, cdb);
    int dummy2 = cdb->SetIndex(
      ET::FixedPixelTypeString, ET::FixedDimension, ET::MovingPixelTypeString, ET::MovingDimension, VIndex);
    if (ElastixTypedef<VIndex + 1>::IsDefined)
    {
      return _installsupportedimagesrecursively<VIndex + 1>::DO(name, cdb);
    }
    return (dummy1 + dummy2);
  }
};

// end template class

/** To prevent an infinite loop, DO() does nothing in class<lastImageTypeCombination> */
template <>
class _installsupportedimagesrecursively<NrOfSupportedImageTypes + 1>
{
public:
  using ComponentDescriptionType = ComponentDatabase::ComponentDescriptionType;
  static int
  DO(const ComponentDescriptionType & /** name */, ComponentDatabase * /** cdb */)
  {
    return 0;
  }
};

// end template class specialization

/**
 * ****************** Constructor ********************************
 */

ComponentLoader::ComponentLoader()
{
  this->m_ImageTypeSupportInstalled = false;
}


/**
 * ****************** Destructor *********************************
 */

ComponentLoader::~ComponentLoader() = default;


/**
 * *************** InstallSupportedImageTypes ********************
 */
int
ComponentLoader::InstallSupportedImageTypes()
{
  /**
   * Method: A recursive template was defined at the top of this file, that
   * installs support for all combinations of ImageTypes defined in
   * elxSupportedImageTypes.h
   *
   * Result: The VIndices are stored in the elx::ComponentDatabase::IndexMap.
   * The New() functions of ElastixTemplate<> in the
   * elx::ComponentDatabase::CreatorMap, with key "Elastix".
   */

  /** Call class<1>::DO(...) */
  int _InstallDummy_SupportedImageTypes =
    _installsupportedimagesrecursively<1>::DO("Elastix", this->m_ComponentDatabase);

  if (_InstallDummy_SupportedImageTypes == 0)
  {
    this->m_ImageTypeSupportInstalled = true;
  }

  return _InstallDummy_SupportedImageTypes;

} // end InstallSupportedImageTypes


/**
 * ****************** LoadComponents *****************************
 */

int
ComponentLoader::LoadComponents()
{
  // Retrieve those IOFactoryRegisterManager instances, just to ensure that they are really constructed.
  const volatile auto ioFactoryRegisterManagerInstances =
    std::make_tuple(itk::ImageIOFactoryRegisterManagerInstance,
                    itk::MeshIOFactoryRegisterManagerInstance,
                    itk::TransformIOFactoryRegisterManagerInstance);
  (void)ioFactoryRegisterManagerInstances;

  int installReturnCode = 0;

  /** Generate the mapping between indices and image types */
  if (!this->m_ImageTypeSupportInstalled)
  {
    installReturnCode = this->InstallSupportedImageTypes();
    if (installReturnCode != 0)
    {
      log::error("ERROR: ImageTypeSupport installation failed. ");
      return installReturnCode;
    }
  } // end if !ImageTypeSupportInstalled

  log::info("Installing all components.");

  /** Fill the component database */
  installReturnCode = InstallAllComponents(this->m_ComponentDatabase);

  if (installReturnCode)
  {
    log::error("ERROR: Installing of at least one of components failed.");
    return installReturnCode;
  }

  log::info("InstallingComponents was successful.\n");

  return 0;

} // end LoadComponents

} // end namespace elastix
