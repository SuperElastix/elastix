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
#ifndef elxElastixMain_h
#define elxElastixMain_h

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

///
/// ********************** Global Functions **********************
///
/// NB: not part of the ElastixMain class.
///

/**
 * function xoutSetup
 * Configure the xl::xout variable, which has to be used for
 * for writing messages. The function adds some default fields,
 * such as "warning", "error", "standard", "logonly" and "coutonly",
 * and it sets the outputs to std::cout and/or a logfile.
 *
 * The method takes a logfile name as its input argument.
 * It returns 0 if everything went ok. 1 otherwise.
 */
extern int
xoutSetup(const char * logfilename, bool setupLogging, bool setupCout);


/** Manages setting up and closing the "xout" output streams.
 */
class xoutManager
{
public:
  ITK_DISALLOW_COPY_AND_ASSIGN(xoutManager);

  /** This explicit constructor does set up the "xout" output streams. */
  explicit xoutManager(const std::string & logfilename, const bool setupLogging, const bool setupCout);

  /** The default-constructor only just constructs a manager object */
  xoutManager() = default;

  /** The destructor closes the "xout" output streams. */
  ~xoutManager() = default;

private:
  struct Guard
  {
    ITK_DISALLOW_COPY_AND_ASSIGN(Guard);
    Guard() = default;
    ~Guard();
  };

  const Guard m_Guard{};
};


/**
 * \class ElastixMain
 * \brief A class with all functionality to configure elastix.
 *
 * The ElastixMain initializes the MyConfiguration class with the
 * parameters and commandline arguments. After this, the class loads
 * and creates all components and sets them in ElastixTemplate.
 *
 * \parameter FixedImageDimension: the dimension of the fixed image. \n
 * example: <tt>(FixedImageDimension 2)</tt>\n
 * \parameter MovingImageDimension: the dimension of the fixed image. \n
 * example: <tt>(MovingImageDimension 2)</tt>\n
 * \parameter FixedInternalImagePixelType: the pixel type of the internal
 * fixed image representation. The fixed image is automatically converted
 * to this type.\n
 * example: <tt>(FixedInternalImagePixelType "float")</tt>\n
 * Default/recommended: "float"\n
 * \parameter MovingInternalImagePixelType: the pixel type of the internal
 * moving image representation. The moving image is automatically converted
 * to this type.\n
 * example: <tt>(MovingInternalImagePixelType "float")</tt>\n
 * Default/recommended: "float"\n
 *
 * \transformparameter FixedImageDimension: the dimension of the fixed image. \n
 * example: <tt>(FixedImageDimension 2)</tt>\n
 * \transformparameter MovingImageDimension: the dimension of the fixed image. \n
 * example: <tt>(MovingImageDimension 2)</tt>\n
 * \transformparameter FixedInternalImagePixelType: the pixel type of the internal
 * fixed image representation. The fixed image is automatically converted
 * to this type.\n
 * example: <tt>(FixedInternalImagePixelType "float")</tt>\n
 * Default/recommended: "float"\n
 * \transformparameter MovingInternalImagePixelType: the pixel type of the internal
 * moving image representation. The moving image is automatically converted
 * to this type.\n
 * example: <tt>(MovingInternalImagePixelType "float")</tt>\n
 * Default/recommended: "float"\n
 *
 * \ingroup Kernel
 */

class ElastixMain : public itk::Object
{
public:
  /** Standard itk. */
  typedef ElastixMain                   Self;
  typedef itk::Object                   Superclass;
  typedef itk::SmartPointer<Self>       Pointer;
  typedef itk::SmartPointer<const Self> ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(ElastixMain, Object);

  /** Typedef's.*/

  /** ITK base objects. */
  typedef itk::Object             ObjectType;
  typedef ObjectType::Pointer     ObjectPointer;
  typedef itk::DataObject         DataObjectType;
  typedef DataObjectType::Pointer DataObjectPointer;

  /** elastix components. */
  typedef ElastixBase                                   ElastixBaseType;
  typedef ElastixBase::ConfigurationType                ConfigurationType;
  typedef ConfigurationType::CommandLineArgumentMapType ArgumentMapType;
  typedef ConfigurationType::Pointer                    ConfigurationPointer;
  typedef ElastixBase::ObjectContainerType              ObjectContainerType;
  typedef ElastixBase::DataObjectContainerType          DataObjectContainerType;
  typedef ElastixBase::ObjectContainerPointer           ObjectContainerPointer;
  typedef ElastixBase::DataObjectContainerPointer       DataObjectContainerPointer;
  typedef ElastixBase::FlatDirectionCosinesType         FlatDirectionCosinesType;

  /** Typedefs for the database that holds pointers to New() functions.
   * Those functions are used to instantiate components, such as the metric etc.
   */
  typedef ComponentDatabase                               ComponentDatabaseType;
  typedef ComponentDatabaseType::Pointer                  ComponentDatabasePointer;
  typedef ComponentDatabaseType::PtrToCreator             PtrToCreator;
  typedef ComponentDatabaseType::ComponentDescriptionType ComponentDescriptionType;
  typedef ComponentDatabaseType::PixelTypeDescriptionType PixelTypeDescriptionType;
  typedef ComponentDatabaseType::ImageDimensionType       ImageDimensionType;
  typedef ComponentDatabaseType::IndexType                DBIndexType;

  /** Typedef that is used in the elastix dll version. */
  typedef itk::ParameterMapInterface::ParameterMapType ParameterMapType;

  /** Set/Get functions for the description of the image type. */
  itkSetMacro(FixedImagePixelType, PixelTypeDescriptionType);
  itkSetMacro(MovingImagePixelType, PixelTypeDescriptionType);
  itkSetMacro(FixedImageDimension, ImageDimensionType);
  itkSetMacro(MovingImageDimension, ImageDimensionType);
  itkGetMacro(FixedImagePixelType, PixelTypeDescriptionType);
  itkGetMacro(MovingImagePixelType, PixelTypeDescriptionType);
  itkGetMacro(FixedImageDimension, ImageDimensionType);
  itkGetMacro(MovingImageDimension, ImageDimensionType);

  /** Set/Get functions for the fixed and moving images
   * (if these are not used, elastix tries to read them from disk,
   * according to the command line parameters).
   */
  itkSetObjectMacro(FixedImageContainer, DataObjectContainerType);
  itkSetObjectMacro(MovingImageContainer, DataObjectContainerType);
  itkGetModifiableObjectMacro(FixedImageContainer, DataObjectContainerType);
  itkGetModifiableObjectMacro(MovingImageContainer, DataObjectContainerType);

  /** Set/Get functions for the fixed and moving masks
   * (if these are not used, elastix tries to read them from disk,
   * according to the command line parameters).
   */
  itkSetObjectMacro(FixedMaskContainer, DataObjectContainerType);
  itkSetObjectMacro(MovingMaskContainer, DataObjectContainerType);
  itkGetModifiableObjectMacro(FixedMaskContainer, DataObjectContainerType);
  itkGetModifiableObjectMacro(MovingMaskContainer, DataObjectContainerType);

  /** Set/Get functions for the result images
   * (if these are not used, elastix tries to read them from disk,
   * according to the command line parameters).
   */
  itkSetObjectMacro(ResultImageContainer, DataObjectContainerType);
  itkGetModifiableObjectMacro(ResultImageContainer, DataObjectContainerType);

  itkSetObjectMacro(ResultDeformationFieldContainer, DataObjectContainerType);
  itkGetModifiableObjectMacro(ResultDeformationFieldContainer, DataObjectContainerType);

  /** Set/Get the configuration object. */
  itkSetObjectMacro(Configuration, ConfigurationType);
  itkGetModifiableObjectMacro(Configuration, ConfigurationType);

  /** Functions to get pointers to the elastix components.
   * The components are returned as Object::Pointer.
   * Before calling this functions, call run().
   */
  itkGetModifiableObjectMacro(Elastix, ObjectType);

  /** Convenience function that returns the elastix component as
   * a pointer to an ElastixBaseType. Use only after having called run()!
   */
  ElastixBaseType &
  GetElastixBase(void) const;

  /** Get the final transform (the result of running elastix).
   * You may pass this as an InitialTransform in an other instantiation
   * of ElastixMain.
   * Only valid after calling Run()!
   */
  itkGetModifiableObjectMacro(FinalTransform, ObjectType);

  /** Set/Get the initial transform
   * the type is ObjectType, but the pointer should actually point
   * to an itk::Transform type (or inherited from that one).
   */
  itkSetObjectMacro(InitialTransform, ObjectType);
  itkGetModifiableObjectMacro(InitialTransform, ObjectType);

  /** Set/Get the original fixed image direction as a flat array
   * (d11 d21 d31 d21 d22 etc ) */
  virtual void
  SetOriginalFixedImageDirectionFlat(const FlatDirectionCosinesType & arg);

  virtual const FlatDirectionCosinesType &
  GetOriginalFixedImageDirectionFlat(void) const;

  /** Get and Set the elastix level. */
  void
  SetElastixLevel(unsigned int level);

  unsigned int
  GetElastixLevel(void);

  /** Get and Set the total number of elastix levels. */
  void
  SetTotalNumberOfElastixLevels(unsigned int levels);

  unsigned int
  GetTotalNumberOfElastixLevels(void);

  /** Returns the Index that is used in elx::ComponentDatabase. */
  itkGetConstMacro(DBIndex, DBIndexType);

  /** Enter the command line parameters, which were given by the user,
   * if elastix.exe is used to do a registration.
   * The Configuration object will be initialized in this way.
   */
  virtual void
  EnterCommandLineArguments(const ArgumentMapType & argmap);

  virtual void
  EnterCommandLineArguments(const ArgumentMapType & argmap, const ParameterMapType & inputMap);

  // Version used when elastix is used as a library.
  virtual void
  EnterCommandLineArguments(const ArgumentMapType & argmap, const std::vector<ParameterMapType> & inputMaps);

  /** Start the registration
   * run() without command line parameters; it assumes that
   * EnterCommandLineParameters has been invoked already, or that
   * m_Configuration is initialized in a different way.
   */
  virtual int
  Run(void);

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
  SetProcessPriority(void) const;

  /** Set maximum number of threads, which is read from the command line arguments.
   * Syntax:
   * -threads \<int\>
   */
  virtual void
  SetMaximumNumberOfThreads(void) const;

  /** Function to get the ComponentDatabase. */
  static const ComponentDatabase &
  GetComponentDatabase(void);

  /** GetTransformParametersMap */
  virtual ParameterMapType
  GetTransformParametersMap(void) const;

protected:
  ElastixMain();
  ~ElastixMain() override;

  /** A pointer to elastix as an itk::object. In run() this
   * pointer will be assigned to an ElastixTemplate<>.
   */
  ObjectPointer m_Elastix;

  /** The configuration object, containing the parameters and command-line arguments. */
  ConfigurationPointer m_Configuration;

  /** A vector of configuration objects, needed when transformix is used as library. */
  std::vector<ConfigurationPointer> m_Configurations;

  /** Description of the ImageTypes. */
  PixelTypeDescriptionType m_FixedImagePixelType;
  ImageDimensionType       m_FixedImageDimension;
  PixelTypeDescriptionType m_MovingImagePixelType;
  ImageDimensionType       m_MovingImageDimension;

  DBIndexType m_DBIndex;

  /** The images and masks. */
  DataObjectContainerPointer m_FixedImageContainer;
  DataObjectContainerPointer m_MovingImageContainer;
  DataObjectContainerPointer m_FixedMaskContainer;
  DataObjectContainerPointer m_MovingMaskContainer;
  DataObjectContainerPointer m_ResultImageContainer;
  DataObjectContainerPointer m_ResultDeformationFieldContainer;

  /** A transform that is the result of registration. */
  ObjectPointer m_FinalTransform;

  /** The initial transform. */
  ObjectPointer m_InitialTransform;
  /** Transformation parameters map containing parameters that is the
   *  result of registration.
   */
  ParameterMapType m_TransformParametersMap;

  FlatDirectionCosinesType m_OriginalFixedImageDirection;

  /** InitDBIndex sets m_DBIndex by asking the ImageTypes
   * from the Configuration object and obtaining the corresponding
   * DB index from the ComponentDatabase.
   */
  virtual int
  InitDBIndex(void);

  /** Create a component. Make sure InitDBIndex has been called before.
   * The input is a string, e.g. "MattesMutualInformation".
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

  /** Helper function to obtain information from images on disk. */
  void
  GetImageInformationFromFile(const std::string & filename, ImageDimensionType & imageDimension) const;

private:
  ElastixMain(const Self &) = delete;
  void
  operator=(const Self &) = delete;
};

} // end namespace elastix

#endif // end #ifndef elxElastixMain_h
