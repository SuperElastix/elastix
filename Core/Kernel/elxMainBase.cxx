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

/** If running on a Windows-system, include "windows.h".
 *  This is to set the priority, but which does not work on cygwin.
 */

#if defined(_WIN32) && !defined(__CYGWIN__)
#  include <windows.h>
#endif

#include "elxMainBase.h"
#include "elxComponentLoader.h"

#include "elxMacro.h"
#include "itkMultiThreaderBase.h"

#ifdef ELASTIX_USE_OPENCL
#  include "itkOpenCLContext.h"
#  include "itkOpenCLSetup.h"
#endif

namespace elastix
{

/**
 * ********************* Constructor ****************************
 */

MainBase::MainBase() = default;

/**
 * ****************** GetComponentDatabase *********
 */

const ComponentDatabase &
MainBase::GetComponentDatabase()
{
  // Improved thread-safety by using C++11 "magic statics".
  static const auto staticComponentDatabase = [] {
    const auto componentDatabase = ComponentDatabase::New();
    const auto componentLoader = ComponentLoader::New();
    componentLoader->SetComponentDatabase(componentDatabase);

    if (componentLoader->LoadComponents() != 0)
    {
      log::error("Loading components failed");
    }
    return componentDatabase;
  }();
  return *staticComponentDatabase;
}


/**
 * ********************** Destructor ****************************
 */

MainBase::~MainBase()
{
#ifdef ELASTIX_USE_OPENCL
  itk::OpenCLContext::Pointer context = itk::OpenCLContext::GetInstance();
  if (context->IsCreated())
  {
    context->Release();
  }
#endif
} // end Destructor


/**
 * **************************** Run *****************************
 */

int
MainBase::Run(const ArgumentMapType & argmap)
{
  /** Initialize the configuration object with the
   * command line parameters entered by the user.
   */
  if (m_Configuration->Initialize(argmap) != 0)
  {
    log::error("ERROR: Something went wrong during initialization of the configuration object.");
  }
  return this->Run();
} // end Run()


/**
 * **************************** Run *****************************
 */

int
MainBase::Run(const ArgumentMapType & argmap, const ParameterMapType & inputMap)
{
  /** Initialize the configuration object with the
   * command line parameters entered by the user.
   */
  if (m_Configuration->Initialize(argmap, inputMap) != 0)
  {
    log::error("ERROR: Something went wrong during initialization of the configuration object.");
  }

  return this->Run();
} // end Run()


/**
 * ************************* GetElastixBase ***************************
 */

ElastixBase &
MainBase::GetElastixBase() const
{
  /** Convert ElastixAsObject to a pointer to an ElastixBase. */
  const auto elastixBase = dynamic_cast<ElastixBase *>(m_Elastix.GetPointer());
  if (elastixBase == nullptr)
  {
    itkExceptionMacro("Probably GetElastixBase() is called before having called Run()");
  }

  return *elastixBase;

} // end GetElastixBase()


/**
 * ************************* CreateComponent ***************************
 */

MainBase::ObjectPointer
MainBase::CreateComponent(const ComponentDescriptionType & name)
{
  /** A pointer to the New() function. */
  if (const PtrToCreator creator = GetComponentDatabase().GetCreator(name, m_DBIndex))
  {
    if (const ObjectPointer component = creator())
    {
      return component;
    }
  }

  itkExceptionMacro("The following component could not be created: " << name);

} // end CreateComponent()


/**
 * *********************** CreateComponents *****************************
 */

MainBase::ObjectContainerPointer
MainBase::CreateComponents(const std::string &              key,
                           const ComponentDescriptionType & defaultComponentName,
                           int &                            errorcode,
                           bool                             mandatoryComponent)
{
  ComponentDescriptionType componentName = defaultComponentName;
  unsigned int             componentnr = 0;
  ObjectContainerPointer   objectContainer = ObjectContainerType::New();
  objectContainer->Initialize();

  /** Read the component name.
   * If the user hasn't specified any component names, use
   * the default, and give a warning.
   */
  bool found = m_Configuration->ReadParameter(componentName, key, componentnr, true);

  /** If the default equals "" (so no default), the mandatoryComponent
   * flag is true, and not component was given by the user,
   * then elastix quits.
   */
  if (!found && (defaultComponentName.empty()))
  {
    if (mandatoryComponent)
    {
      log::error(std::ostringstream{} << "ERROR: the following component has not been specified: " << key);
      errorcode = 1;
      return objectContainer;
    }
    else
    {
      /* Just return an empty container without nagging. */
      errorcode = 0;
      return objectContainer;
    }
  }

  /** Try creating the specified component. */
  try
  {
    objectContainer->CreateElementAt(componentnr) = this->CreateComponent(componentName);
  }
  catch (const itk::ExceptionObject & excp)
  {
    log::error(std::ostringstream{} << "ERROR: error occurred while creating " << key << " " << componentnr << ".\n"
                                    << excp);
    errorcode = 1;
    return objectContainer;
  }

  /** Check if more than one component name is given. */
  while (found)
  {
    ++componentnr;
    found = m_Configuration->ReadParameter(componentName, key, componentnr, false);
    if (found)
    {
      try
      {
        objectContainer->CreateElementAt(componentnr) = this->CreateComponent(componentName);
      }
      catch (const itk::ExceptionObject & excp)
      {
        log::error(std::ostringstream{} << "ERROR: error occurred while creating " << key << " " << componentnr << ".\n"
                                        << excp);
        errorcode = 1;
        return objectContainer;
      }
    } // end if
  }   // end while

  return objectContainer;

} // end CreateComponents()


/**
 * *********************** SetProcessPriority *************************
 */

void
MainBase::SetProcessPriority() const
{
  /** If wanted, set the priority of this process high or below normal. */
  std::string processPriority = m_Configuration->GetCommandLineArgument("-priority");
  if (processPriority == "high")
  {
#if defined(_WIN32) && !defined(__CYGWIN__)
    SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);
#endif
  }
  else if (processPriority == "abovenormal")
  {
#if defined(_WIN32) && !defined(__CYGWIN__)
    SetPriorityClass(GetCurrentProcess(), ABOVE_NORMAL_PRIORITY_CLASS);
#endif
  }
  else if (processPriority == "normal")
  {
#if defined(_WIN32) && !defined(__CYGWIN__)
    SetPriorityClass(GetCurrentProcess(), NORMAL_PRIORITY_CLASS);
#endif
  }
  else if (processPriority == "belownormal")
  {
#if defined(_WIN32) && !defined(__CYGWIN__)
    SetPriorityClass(GetCurrentProcess(), BELOW_NORMAL_PRIORITY_CLASS);
#endif
  }
  else if (processPriority == "idle")
  {
#if defined(_WIN32) && !defined(__CYGWIN__)
    SetPriorityClass(GetCurrentProcess(), IDLE_PRIORITY_CLASS);
#endif
  }
  else if (!processPriority.empty())
  {
    log::warn("Unsupported -priority value. Specify one of <high, abovenormal, normal, belownormal, idle, ''>.");
  }

} // end SetProcessPriority()


/**
 * *********************** SetMaximumNumberOfThreads *************************
 */

void
MainBase::SetMaximumNumberOfThreads() const
{
  /** Get the number of threads from the command line. */
  std::string maximumNumberOfThreadsString = m_Configuration->GetCommandLineArgument("-threads");

  /** If supplied, set the maximum number of threads. */
  if (!maximumNumberOfThreadsString.empty())
  {
    const int maximumNumberOfThreads = atoi(maximumNumberOfThreadsString.c_str());
    itk::MultiThreaderBase::SetGlobalMaximumNumberOfThreads(maximumNumberOfThreads);

    // The following statement (getting and setting GlobalDefaultNumberOfThreads) may look redundant, but it's not
    // (using ITK 5.4.0)! The Set function ensures that GlobalDefaultNumberOfThreads <= GlobalMaximumNumberOfThreads.
    // (GlobalDefaultNumberOfThreads is important, as ITK uses this number when constructing the ThreadPool.)
    itk::MultiThreaderBase::SetGlobalDefaultNumberOfThreads(itk::MultiThreaderBase::GetGlobalDefaultNumberOfThreads());
  }
} // end SetMaximumNumberOfThreads()

} // end namespace elastix
