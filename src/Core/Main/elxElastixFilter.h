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

  /** Standard ITK typedefs. */
  typedef ElastixFilter                   Self;
  typedef itk::ImageSource< TFixedImage > Superclass; 
  typedef itk::SmartPointer< Self >       Pointer;
  typedef itk::SmartPointer< const Self > ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro( Self );

  /** Run-time type information (and related methods). */
  itkTypeMacro( Self, itk::ImageSource );

  /** Typedefs. */
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

  // MS: \todo: see if you can use SetConstObjectMacro's and GetConstObjectMacro's below.
  //            have a look at the signatures of the itk::ImageToImageMetric functions and copy them if possible.
  // MS: \todo: add get macro's
  // MS: \todo: move all implementations to the hxx file

  /** Set/Get/Add fixed image. */
  void SetFixedImage( FixedImagePointer fixedImage );
  void SetFixedImage( DataObjectContainerPointer fixedImages );
  void AddFixedImage( FixedImagePointer fixedImage );

  /** Set/Get/Add moving image. */
  void SetMovingImage( MovingImagePointer movingImages );
  void SetMovingImage( DataObjectContainerPointer movingImages );
  void AddMovingImage( MovingImagePointer movingImage );

  /** Set/Get/Add fixed mask. */
  void SetFixedMask( FixedMaskPointer fixedMask );
  void SetFixedMask( DataObjectContainerPointer fixedMasks );
  void AddFixedMask( FixedMaskPointer fixedMask );
  void RemoveFixedMask( void );

  /** Set/Get/Add moving mask. */
  void SetMovingMask( MovingMaskPointer movingMask );
  void SetMovingMask( DataObjectContainerPointer movingMasks );
  void AddMovingMask( MovingMaskPointer movingMask );
  void RemoveMovingMask( void );

  /** Set/Get parameter object.*/
  void SetParameterObject( ParameterObjectPointer parameterObject );
  ParameterObjectPointer GetParameterObject( void );
  ParameterObjectPointer GetTransformParameterObject( void );

  /** Set/Get/Remove initial transform parameter filename. */
  // TODO: Pass transform object instead of reading from disk
  itkSetMacro( InitialTransformParameterFileName, std::string );
  itkGetConstReferenceMacro( InitialTransformParameterFileName, std::string );
  void RemoveInitialTransformParameterFileName( void ) { this->SetInitialTransformParameterFileName( "" ); };

  /** Set/Get/Remove fixed point set filename. */
  itkSetMacro( FixedPointSetFileName, std::string );
  itkGetConstReferenceMacro( FixedPointSetFileName, std::string );
  void RemoveFixedPointSetFileName( void ) { this->SetFixedPointSetFileName( "" ); };

  /** Set/Get/Remove moving point set filename. */
  itkSetMacro( MovingPointSetFileName, std::string );
  itkGetConstReferenceMacro( MovingPointSetFileName, std::string );
  void RemoveMovingPointSetFileName( void ) { this->SetMovingPointSetFileName( "" ); };

  /** Set/Get/Remove output directory. */
  itkSetMacro( OutputDirectory, std::string );
  itkGetConstReferenceMacro( OutputDirectory, std::string );
  void RemoveOutputDirectory() { this->SetOutputDirectory( "" ); };

  /** Set/Get/Remove log filename. */
  void SetLogFileName( std::string logFileName );
  itkGetConstMacro( LogFileName, std::string );
  void RemoveLogFileName( void );

  /** Log to std::cout on/off. */
  itkSetMacro( LogToConsole, bool );
  itkGetConstReferenceMacro( LogToConsole, bool );
  itkBooleanMacro( LogToConsole );

  /** Log to file on/off. */
  itkSetMacro( LogToFile, bool );
  itkGetConstReferenceMacro( LogToFile, bool );
  itkBooleanMacro( LogToFile );

protected:

  ElastixFilter( void );

  void GenerateData( void ) ITK_OVERRIDE;

private: 

  ElastixFilter( const Self & );  // purposely not implemented
  void operator=( const Self & ); // purposely not implemented

  /** SetInputWithUniqueName. */
  void SetInputWithUniqueName( const DataObjectIdentifierType& key, itk::DataObject* input );

  /** IsInputType. */
  bool IsInputType( const DataObjectIdentifierType& inputType, DataObjectIdentifierType inputName );

  /** RemoveInputType. */
  void RemoveInputType( const DataObjectIdentifierType& inputName );

  /** Let elastix handle input verification internally */
  void VerifyInputInformation( void ) ITK_OVERRIDE {};

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
