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
 * \class ElastixFilter
 * \brief ITK Filter interface to the Elastix registration library.
 */

namespace elastix
{

template< typename TFixedImage, typename TMovingImage >
class ELASTIXLIB_API ElastixFilter : public itk::ImageSource< TFixedImage >
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
  typedef elastix::ElastixMain                      ElastixMainType;
  typedef ElastixMainType::Pointer                  ElastixMainPointer;
  typedef std::vector< ElastixMainPointer >         ElastixMainVectorType;
  typedef ElastixMainType::ObjectPointer            ElastixMainObjectPointer;
  typedef ElastixMainType::ArgumentMapType          ArgumentMapType;
  typedef ArgumentMapType::value_type               ArgumentMapEntryType;
  typedef ElastixMainType::FlatDirectionCosinesType FlatDirectionCosinesType;

  typedef ElastixMainType::DataObjectContainerType           DataObjectContainerType;
  typedef ElastixMainType::DataObjectContainerPointer        DataObjectContainerPointer;
  typedef DataObjectContainerType::Iterator                  DataObjectContainerIterator;
  typedef itk::ProcessObject::DataObjectIdentifierType       DataObjectIdentifierType;
  typedef itk::ProcessObject::DataObjectPointerArraySizeType DataObjectPointerArraySizeType;
  typedef itk::ProcessObject::NameArray                      NameArrayType;

  typedef ParameterObject                               ParameterObjectType;
  typedef ParameterObjectType::ParameterMapType         ParameterMapType;
  typedef ParameterObjectType::ParameterMapVectorType   ParameterMapVectorType;
  typedef ParameterObjectType::ParameterValueVectorType ParameterValueVectorType;
  typedef ParameterObjectType::Pointer                  ParameterObjectPointer;
  typedef ParameterObjectType::ConstPointer             ParameterObjectConstPointer;

  typedef typename TFixedImage::Pointer       FixedImagePointer;
  typedef typename TFixedImage::ConstPointer  FixedImageConstPointer;
  typedef typename TMovingImage::Pointer      MovingImagePointer;
  typedef typename TMovingImage::ConstPointer MovingImageConstPointer;

  itkStaticConstMacro( FixedImageDimension, unsigned int, TFixedImage::ImageDimension );
  itkStaticConstMacro( MovingImageDimension, unsigned int, TMovingImage::ImageDimension );
  itkGetStaticConstMacro( FixedImageDimension );
  itkGetStaticConstMacro( MovingImageDimension );

  typedef itk::Image< unsigned char, FixedImageDimension >  FixedMaskType;
  typedef typename FixedMaskType::Pointer                   FixedMaskPointer;
  typedef typename FixedMaskType::Pointer                   FixedMaskConstPointer;
  typedef itk::Image< unsigned char, MovingImageDimension > MovingMaskType;
  typedef typename MovingMaskType::Pointer                  MovingMaskPointer;
  typedef typename MovingMaskType::Pointer                  MovingMaskConstPointer;

  /** Set/Add/Get/NumberOf fixed images. */
  virtual void SetFixedImage( TFixedImage * fixedImage );
  virtual void AddFixedImage( TFixedImage * fixedImage );
  FixedImageConstPointer GetFixedImage( void ) const;
  FixedImageConstPointer GetFixedImage( const unsigned int index ) const;
  unsigned int GetNumberOfFixedImages( void ) const;

  /** Set/Add/Get/NumberOf moving images. */
  virtual void SetMovingImage( TMovingImage * movingImages );
  virtual void AddMovingImage( TMovingImage * movingImage );
  MovingImageConstPointer GetMovingImage( void ) const;
  MovingImageConstPointer GetMovingImage( const unsigned int index ) const;
  unsigned int GetNumberOfMovingImages( void ) const;

  /** Set/Add/Get/Remove/NumberOf fixed masks. */
  virtual void AddFixedMask( FixedMaskType * fixedMask );
  virtual void SetFixedMask( FixedMaskType * fixedMask );
  FixedMaskConstPointer GetFixedMask( void ) const;
  FixedMaskConstPointer GetFixedMask( const unsigned int index ) const;
  void RemoveFixedMask( void );
  unsigned int GetNumberOfFixedMasks( void ) const;

  /** Set/Add/Get/Remove/NumberOf moving masks. */
  virtual void SetMovingMask( MovingMaskType * movingMask );
  virtual void AddMovingMask( MovingMaskType * movingMask );
  MovingMaskConstPointer GetMovingMask( void ) const;
  MovingMaskConstPointer GetMovingMask( const unsigned int index ) const;
  virtual void RemoveMovingMask( void );
  unsigned int GetNumberOfMovingMasks( void ) const;

  /** Set/Get parameter object.*/
  virtual void SetParameterObject( ParameterObjectType * parameterObject );
  ParameterObjectType * GetParameterObject( void );
  const ParameterObjectType * GetParameterObject( void ) const;

  /** Get transform parameter object.*/
  ParameterObjectType * GetTransformParameterObject( void );
  const ParameterObjectType * GetTransformParameterObject( void ) const;

  /** Set/Get/Remove initial transform parameter filename. */
  itkSetMacro( InitialTransformParameterFileName, std::string );
  itkGetMacro( InitialTransformParameterFileName, std::string );
  virtual void RemoveInitialTransformParameterFileName( void ) { this->SetInitialTransformParameterFileName( "" ); }

  /** Set/Get/Remove fixed point set filename. */
  itkSetMacro( FixedPointSetFileName, std::string );
  itkGetMacro( FixedPointSetFileName, std::string );
  void RemoveFixedPointSetFileName( void ) { this->SetFixedPointSetFileName( "" ); }

  /** Set/Get/Remove moving point set filename. */
  itkSetMacro( MovingPointSetFileName, std::string );
  itkGetMacro( MovingPointSetFileName, std::string );
  void RemoveMovingPointSetFileName( void ) { this->SetMovingPointSetFileName( "" ); }

  /** Set/Get/Remove output directory. */
  itkSetMacro( OutputDirectory, std::string );
  itkGetMacro( OutputDirectory, std::string );
  void RemoveOutputDirectory() { this->SetOutputDirectory( "" ); }

  /** Set/Get/Remove log filename. */
  void SetLogFileName( const std::string logFileName );

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

  virtual void GenerateData( void ) ITK_OVERRIDE;

private:

  ElastixFilter( const Self & );  // purposely not implemented
  void operator=( const Self & ); // purposely not implemented

  /** MakeUniqueName. */
  std::string MakeUniqueName( const DataObjectIdentifierType & key );

  /** IsInputOfType. */
  bool IsInputOfType( const DataObjectIdentifierType & InputOfType, DataObjectIdentifierType inputName );

  /** GetNumberOfInputsOfType */
  unsigned int GetNumberOfInputsOfType( const DataObjectIdentifierType & intputType );

  /** RemoveInputsOfType. */
  void RemoveInputsOfType( const DataObjectIdentifierType & inputName );

  /** Let elastix handle input verification internally */
  virtual void VerifyInputInformation( void ) ITK_OVERRIDE {};

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
