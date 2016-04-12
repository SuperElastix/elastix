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
#ifndef elxElastixFilter_h
#define elxElastixFilter_h

#include "itkImageSource.h"

#include "elxElastixMain.h"
#include "elxParameterObject.h"
#include "elxPixelType.h"

/**
 * Elastix registration library exposed as an ITK filter.
 */

namespace elastix
{

template< typename TFixedImage, typename TMovingImage >
class ElastixFilter : public itk::ImageSource< TFixedImage >
{
public:

  typedef ElastixFilter                   Self;
  // typedef itk::ImageSource< TFixedImage > Superclass; 
  typedef itk::SmartPointer< Self >       Pointer;
  typedef itk::SmartPointer< const Self > ConstPointer;
  itkNewMacro( Self );
  itkTypeMacro( Self, itk::ImageSource );

  typedef elastix::ElastixMain                                ElastixMainType;
  typedef ElastixMainType::Pointer                            ElastixMainPointer;
  typedef std::vector< ElastixMainPointer >                   ElastixMainVectorType;
  typedef ElastixMainType::ObjectPointer                      ElastixMainObjectPointer;

  typedef ElastixMainType::FlatDirectionCosinesType           FlatDirectionCosinesType;

  typedef ElastixMainType::ArgumentMapType                    ArgumentMapType;
  typedef ArgumentMapType::value_type                         ArgumentMapEntryType;

  typedef ElastixMainType::DataObjectContainerType            DataObjectContainerType;
  typedef ElastixMainType::DataObjectContainerPointer         DataObjectContainerPointer;
  typedef DataObjectContainerType::Iterator                   DataObjectContainerIterator;
  typedef itk::ProcessObject::DataObjectIdentifierType        DataObjectIdentifierType;
  typedef itk::ProcessObject::DataObjectPointerArraySizeType  DataObjectPointerArraySizeType;
  typedef itk::ProcessObject::NameArray                       InputNameArrayType;

  typedef ParameterObject::ParameterMapType                   ParameterMapType;
  typedef ParameterObject::ParameterMapVectorType             ParameterMapVectorType;
  typedef ParameterObject::ParameterValueVectorType           ParameterValueVectorType;
  typedef ParameterObject::Pointer                            ParameterObjectPointer;
  typedef ParameterObject::ConstPointer                       ParameterObjectConstPointer;

  typedef typename TFixedImage::Pointer                       FixedImagePointer;
  typedef typename TMovingImage::Pointer                      MovingImagePointer;

  itkStaticConstMacro( FixedImageDimension, unsigned int, TFixedImage::ImageDimension );
  itkStaticConstMacro( MovingImageDimension, unsigned int, TMovingImage::ImageDimension );

  typedef itk::Image< unsigned char, FixedImageDimension >    FixedMaskType;
  typedef typename FixedMaskType::Pointer                     FixedMaskPointer;
  typedef itk::Image< unsigned char, MovingImageDimension >   MovingMaskType;
  typedef typename MovingMaskType::Pointer                    MovingMaskPointer;

  void SetFixedImage( FixedImagePointer fixedImage );
  void SetFixedImage( DataObjectContainerPointer fixedImages );
  void AddFixedImage( FixedImagePointer fixedImage );

  void SetMovingImage( MovingImagePointer movingImages );
  void SetMovingImage( DataObjectContainerPointer movingImages );
  void AddMovingImage( MovingImagePointer movingImage );

  void SetFixedMask( FixedMaskPointer fixedMask );
  void SetFixedMask( DataObjectContainerPointer fixedMasks );
  void AddFixedMask( FixedMaskPointer fixedMask );
  void RemoveFixedMask( void );

  void SetMovingMask( MovingMaskPointer movingMask );
  void SetMovingMask( DataObjectContainerPointer movingMasks );
  void AddMovingMask( MovingMaskPointer movingMask );
  void RemoveMovingMask( void );

  void SetParameterObject( ParameterObjectPointer parameterObject );
  ParameterObjectPointer GetParameterObject( void );
  ParameterObjectPointer GetTransformParameterObject( void );

  // TODO: Pass transform object instead of reading from disk
  itkSetMacro( InitialTransformParameterFileName, std::string );
  itkGetConstMacro( InitialTransformParameterFileName, std::string );
  void RemoveInitialTransformParameterFileName( void ) { this->SetInitialTransformParameterFileName( "" ); };

  itkSetMacro( FixedPointSetFileName, std::string );
  itkGetConstMacro( FixedPointSetFileName, std::string );
  void RemoveFixedPointSetFileName( void ) { this->SetFixedPointSetFileName( "" ); };

  itkSetMacro( MovingPointSetFileName, std::string );
  itkGetConstMacro( MovingPointSetFileName, std::string );
  void RemoveMovingPointSetFileName( void ) { this->SetMovingPointSetFileName( "" ); };

  itkSetMacro( OutputDirectory, std::string );
  itkGetConstMacro( OutputDirectory, std::string );
  void RemoveOutputDirectory() { this->SetOutputDirectory( "" ); };

  void SetLogFileName( std::string logFileName )
  {
    this->m_LogFileName = logFileName;
    this->LogToFileOn();
  }

  itkGetConstMacro( LogFileName, std::string );

  void RemoveLogFileName( void ) {
    this->SetLogFileName( "" );
    this->LogToFileOff();
  };

  itkSetMacro( LogToConsole, bool );
  itkGetConstMacro( LogToConsole, bool );
  itkBooleanMacro( LogToConsole );

  itkSetMacro( LogToFile, bool );
  itkGetConstMacro( LogToFile, bool );
  itkBooleanMacro( LogToFile );

  // TODO: We should not have to override GetOutput from superclass. This is a bug.
  // FixedImagePointer GetOutput( void );

protected:

  void GenerateData( void ) ITK_OVERRIDE;

private:

  // using itk::ProcessObject::GetInput;

  ElastixFilter( void );

  ElastixFilter( const Self & );
  void operator=( const Self & );

  void SetInputWithUniqueName( DataObjectIdentifierType key, itk::DataObject* input );
  bool IsInputType( DataObjectIdentifierType inputType, DataObjectIdentifierType inputName );
  void RemoveInputType( DataObjectIdentifierType inputName );
  bool IsEmpty( FixedImagePointer image );

  // Let elastix handle input verification internally
  void VerifyInputInformation( void ) ITK_OVERRIDE {};

  // TODO: When set to true, ReleaseDataFlag should also touch these containers
  DataObjectContainerPointer m_FixedImageContainer;
  DataObjectContainerPointer m_MovingImageContainer;
  DataObjectContainerPointer m_FixedMaskContainer;
  DataObjectContainerPointer m_MovingMaskContainer;

  std::string m_InitialTransformParameterFileName;
  std::string m_FixedPointSetFileName;
  std::string m_MovingPointSetFileName;

  std::string m_OutputDirectory;
  std::string m_LogFileName;

  bool m_LogToConsole;
  bool m_LogToFile;

  unsigned int m_InputUID;

};

} // namespace elx

#ifndef ITK_MANUAL_INSTANTIATION
#include "elxElastixFilter.hxx"
#endif

#endif // elxElastixFilter_h
