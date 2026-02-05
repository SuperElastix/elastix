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
#ifndef elxMainBase_h
#define elxMainBase_h

#include "elxComponentDatabase.h"

#include "elxElastixBase.h"
#include "itkParameterMapInterface.h"

#include <itkObject.h>

// Standard C++ header files:
#include <fstream>
#include <iostream>
#include <string>


namespace elastix
{

/**
 * \class MainBase
 * \brief Common (abstract) base class of ElastixMain and TransformixMain.
 *
 * \ingroup Kernel
 */

class MainBase : public itk::Object
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(MainBase);

  /** Standard itk. */
  using Self = MainBase;
  using Superclass = itk::Object;
  using Pointer = itk::SmartPointer<Self>;
  using ConstPointer = itk::SmartPointer<const Self>;

  /** Run-time type information (and related methods). */
  itkOverrideGetNameOfClassMacro(MainBase);

  /** Typedef's.*/

  /** ITK base objects. */
  using ObjectPointer = itk::Object::Pointer;
  using DataObjectPointer = itk::DataObject::Pointer;

  /** elastix components. */
  using ArgumentMapType = Configuration::CommandLineArgumentMapType;
  using ObjectContainerType = ElastixBase::ObjectContainerType;
  using DataObjectContainerType = ElastixBase::DataObjectContainerType;
  using ObjectContainerPointer = ElastixBase::ObjectContainerPointer;
  using DataObjectContainerPointer = ElastixBase::DataObjectContainerPointer;
  using FlatDirectionCosinesType = ElastixBase::FlatDirectionCosinesType;

  /** Typedefs for the database that holds pointers to New() functions.
   * Those functions are used to instantiate components, such as the metric etc.
   */
  using ComponentDatabasePointer = ComponentDatabase::Pointer;
  using PtrToCreator = ComponentDatabase::PtrToCreator;
  using ComponentDescriptionType = ComponentDatabase::ComponentDescriptionType;
  using PixelTypeDescriptionType = ComponentDatabase::PixelTypeDescriptionType;
  using ImageDimensionType = ComponentDatabase::ImageDimensionType;
  using DBIndexType = ComponentDatabase::IndexType;

  /** Typedef that is used in the elastix dll version. */
  using ParameterMapType = itk::ParameterMapInterface::ParameterMapType;

  /** Set/Get functions for the description of the image type. */
  itkSetMacro(FixedImagePixelType, PixelTypeDescriptionType);
  itkSetMacro(MovingImagePixelType, PixelTypeDescriptionType);
  itkSetMacro(FixedImageDimension, ImageDimensionType);
  itkSetMacro(MovingImageDimension, ImageDimensionType);
  itkGetConstMacro(FixedImagePixelType, PixelTypeDescriptionType);
  itkGetConstMacro(MovingImagePixelType, PixelTypeDescriptionType);
  itkGetConstMacro(FixedImageDimension, ImageDimensionType);
  itkGetConstMacro(MovingImageDimension, ImageDimensionType);

  /** Set/Get functions for the moving images
   * (if these are not used, elastix tries to read them from disk,
   * according to the command line parameters).
   */
  itkSetObjectMacro(MovingImageContainer, DataObjectContainerType);
  itkGetModifiableObjectMacro(MovingImageContainer, DataObjectContainerType);

  /** Set/Get functions for the result images
   * (if these are not used, elastix tries to read them from disk,
   * according to the command line parameters).
   */
  itkSetObjectMacro(ResultImageContainer, DataObjectContainerType);
  itkGetModifiableObjectMacro(ResultImageContainer, DataObjectContainerType);

  itkSetObjectMacro(ResultDeformationFieldContainer, DataObjectContainerType);
  itkGetModifiableObjectMacro(ResultDeformationFieldContainer, DataObjectContainerType);

  /** Set/Get the configuration object. */
  itkSetObjectMacro(Configuration, Configuration);
  itkGetModifiableObjectMacro(Configuration, Configuration);

  /** Functions to get pointers to the elastix components.
   * The components are returned as Object::Pointer.
   * Before calling this functions, call run().
   */
  itkGetModifiableObjectMacro(Elastix, itk::Object);

  /** Convenience function that returns the elastix component as
   * a pointer to an ElastixBase. Use only after having called run()!
   */
  ElastixBase &
  GetElastixBase() const;

  /** Returns the Index that is used in elx::ComponentDatabase. */
  itkGetConstMacro(DBIndex, DBIndexType);

  /** Start the registration
   * run() without command line parameters; it assumes that
   * EnterCommandLineParameters has been invoked already, or that
   * m_Configuration is initialized in a different way.
   */
  virtual int
  Run() = 0;

  /** Start the registration
   * this version of 'run' first calls this->EnterCommandLineParameters(argc,argv)
   * and then calls run().
   */
  virtual int
  Run(const ArgumentMapType & argmap);

  virtual int
  Run(const ArgumentMapType & argmap, const ParameterMapType & inputMap);

  /** Set process priority, which is read from the command line arguments.
   * Syntax:
   * -priority \<high, belownormal\>
   */
  virtual void
  SetProcessPriority() const;

  /** Set maximum number of threads, which is read from the command line arguments.
   * Syntax:
   * -threads \<int\>
   */
  virtual void
  SetMaximumNumberOfThreads() const;


  /**
   * Const accessor for the ComponentDatabase.
   *
   * Most of elastix only needs *read-only* access to the database
   * (querying creators). This overload preserves const-correctness while
   * still sharing the same underlying singleton instance.
   */
  static const ComponentDatabase &
  GetComponentDatabase();

  /**
   * Tries to load a component as a runtime plugin and register it in the
   * ComponentDatabase.
   *
   * This function is called when a component cannot be created because it is
   * not yet present in the ComponentDatabase. It attempts to dynamically load
   * a shared library whose name is derived from `componentName`
   * (e.g. `libImpactMetric.so`, `ImpactMetric.dll`, …), locate the exported
   * `<PluginName>InstallComponent` symbol inside that library, and call it.
   *
   * The install function is expected to register one or more components into
   * the provided ComponentDatabase. After a successful call, the caller can
   * retry component creation using the same name.
   *
   * The plugin is searched using the system dynamic loader mechanisms
   * (RPATH/RUNPATH, LD_LIBRARY_PATH on Unix, PATH on Windows), exactly like a
   * regular shared library.
   *
   * \param componentName  Name of the component as used in the parameter file
   *                       (e.g. "Impact").
   * \return true if a plugin was successfully loaded and installed, false
   *         otherwise.
   */
  bool
  TryLoadComponentPlugin(const ComponentDescriptionType & componentName);

protected:
  MainBase();
  ~MainBase() override = 0;

  /** A pointer to elastix as an itk::object. In run() this
   * pointer will be assigned to an ElastixTemplate<>.
   */
  ObjectPointer m_Elastix{ nullptr };

  /** A vector of configuration objects, needed when transformix is used as library. */
  std::vector<Configuration::ConstPointer> m_TransformConfigurations{};

  /** Description of the ImageTypes. */
  PixelTypeDescriptionType m_FixedImagePixelType{};
  ImageDimensionType       m_FixedImageDimension{ 0 };
  PixelTypeDescriptionType m_MovingImagePixelType{};
  ImageDimensionType       m_MovingImageDimension{ 0 };

  DBIndexType m_DBIndex{ 0 };

  /** The images and masks. */
  DataObjectContainerPointer m_MovingImageContainer{ nullptr };
  DataObjectContainerPointer m_ResultImageContainer{ nullptr };
  DataObjectContainerPointer m_ResultDeformationFieldContainer{ nullptr };

  /** InitDBIndex sets m_DBIndex by asking the ImageTypes
   * from the Configuration object and obtaining the corresponding
   * DB index from the ComponentDatabase.
   */
  virtual int
  InitDBIndex() = 0;

  /**
   * Creates an elastix component by name. Make sure InitDBIndex has been called
   * before invoking this function.
   *
   * The input is a string identifying the component, for example:
   *   "MattesMutualInformation" or "Impact".
   *
   * Component creation follows a two-step strategy:
   *
   *  1) Fast path — static components
   *     First, the function queries the already populated ComponentDatabase.
   *     This covers all components that are statically linked into elastix and
   *     is the normal, zero-overhead code path.
   *
   *  2) Lazy plugin path — optional components
   *     If the component is not found, elastix attempts to *lazy-load* a plugin
   *     that may provide it (for example, the IMPACT metric). When the plugin
   *     is successfully loaded, it registers its components into the database,
   *     and the lookup is retried exactly once.
   *
   * This design allows elastix to:
   *  - run out of the box without heavy optional dependencies (e.g. LibTorch),
   *  - keep startup time minimal,
   *  - enable optional components only when they are actually requested.
   *
   * If both attempts fail, an exception is thrown with a clear diagnostic
   * message explaining which component could not be created.
   */
  virtual ObjectPointer
  CreateComponent(const ComponentDescriptionType & name);

  /** Create components. Reads from the configuration object (using the provided key)
   * the names of the components to create and store their instantiations in the
   * provided ObjectContainer.
   * The errorcode remains what it was if no error occurred. Otherwise it's set to 1.
   * The 'key' is the entry inspected in the parameter file
   * A component named 'defaultComponentName' is used when the key is not found
   * in the parameter file. If "" is used, no default is assumed, and an error
   * is given when the component was not specified. If the flag mandatoryComponent
   * is set to false, no error is given, because the component may not be needed
   * anyway.
   *
   * NB: this function should never be called with:
   * ( !mandatoryComponent && defaultComponentName != "" ) == true
   *
   */
  virtual ObjectContainerPointer
  CreateComponents(const ComponentDescriptionType & key,
                   const ComponentDescriptionType & defaultComponentName,
                   int &                            errorcode,
                   bool                             mandatoryComponent = true);


  /**
   * ****************** GetComponentDatabase *********
   *
   * Returns the global ComponentDatabase instance.
   *
   * This function provides *mutable* access to the database. It is required for
   * the lazy plugin mechanism: when a plugin is loaded at runtime, it must be
   * able to *register new components* into the database. That registration
   * modifies the database, hence a non-const reference is needed.
   *
   * The database itself is created exactly once using a C++11 "magic static".
   * This guarantees:
   *  - thread-safe initialization,
   *  - a single shared ComponentDatabase for the whole process,
   *  - identical behavior to the previous static-initialization design.
   *
   * At construction time, the ComponentLoader installs all statically linked
   * components. Later, TryLoadComponentPlugin() may add more entries by calling
   * into this same mutable instance.
   */
  static ComponentDatabase &
  GetComponentDatabaseMutable();

private:
  /** The configuration object, containing the parameters and command-line arguments. */
  Configuration::Pointer m_Configuration{ Configuration::New() };
};

} // end namespace elastix

#endif // end #ifndef elxMainBase_h
