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

#ifndef elxComponentLoader_h
#define elxComponentLoader_h

#include "elxComponentDatabase.h"
#include "xoutmain.h"

namespace elastix
{

/**
 * \class ComponentLoader
 *
 * \brief Determines which components (metrics, transforms, etc.) are available.
 *
 * This file defines the class elx::ComponentLoader. This class
 * stores pointers to the New() functions of
 * each component in the elx::ComponentDatabase.
 *
 * Each new component (a new metric for example should "make itself
 * known" by calling the elxInstallMacro, which is defined in elxMacro.h.
 */

class ComponentLoader : public itk::Object
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(ComponentLoader);

  /** Standard ITK typedef's. */
  using Self = ComponentLoader;
  using Superclass = itk::Object;
  using Pointer = itk::SmartPointer<Self>;
  using ConstPointer = itk::SmartPointer<const Self>;

  /** Standard ITK stuff. */
  itkNewMacro(Self);
  itkTypeMacro(ComponentLoader, Object);

  /** Typedef's. */
  using ComponentDatabasePointer = ComponentDatabase::Pointer;

  /** Set and get the ComponentDatabase. */
  itkSetObjectMacro(ComponentDatabase, ComponentDatabase);
  itkGetModifiableObjectMacro(ComponentDatabase, ComponentDatabase);

  /** Function to load components. */
  int
  LoadComponents();

protected:
  /** Standard constructor and destructor. */
  ComponentLoader();
  ~ComponentLoader() override;

  ComponentDatabasePointer m_ComponentDatabase;

  bool m_ImageTypeSupportInstalled;
  virtual int
  InstallSupportedImageTypes();

private:
  /** Standard private (copy)constructor. */
};

} // end namespace elastix

#endif // #ifndef elxComponentLoader_h
