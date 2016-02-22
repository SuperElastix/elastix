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
#ifndef __itkNDImageTemplate_h
#define __itkNDImageTemplate_h

#include "itkNDImageBase.h"
#include "itkImageFileReader.h"

namespace itk
{

/**
 * \class NDImageTemplate
 * \brief This class is a specialization of the NDImageBase,
 * which acts as a wrap around an itk::Image.
 *
 * The NDImageTemplate class is a kind of wrap around the
 * itk::Image. It has an itk::Image object as an internal
 * member variable. Most functions simply call the
 * the corresponding function in the itk::Object. For some
 * functions, the in/output arguments have to be converted
 * from/to arrays with runtime length to/from arrays with
 * compile time length.
 *
 * \sa NDImageBase
 * \ingroup Miscellaneous
 */

template< class TPixel, unsigned int VDimension >
class NDImageTemplate : public NDImageBase< TPixel >
{
public:

  /** Standard class typedefs.*/
  typedef NDImageTemplate            Self;
  typedef NDImageBase< TPixel >      Superclass;
  typedef SmartPointer< Self >       Pointer;
  typedef SmartPointer< const Self > ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro( Self );

  /** Run-time type information (and related methods). */
  itkTypeMacro( NDImageTemplate, NDImageBase );

  /**
   * Typedefs inherited from Superclass.
   */

  itkStaticConstMacro( Dimension, unsigned int, VDimension );

  typedef typename Superclass::DataObjectType    DataObjectType;
  typedef typename Superclass::DataObjectPointer DataObjectPointer;

  /** Type definitions like normal itkImages, independent of the dimension */
  typedef typename Superclass::PixelType                  PixelType;
  typedef typename Superclass::ValueType                  ValueType;
  typedef typename Superclass::InternalPixelType          InternalPixelType;
  typedef typename Superclass::AccessorType               AccessorType;
  typedef typename Superclass::PixelContainer             PixelContainer;
  typedef typename Superclass::PixelContainerPointer      PixelContainerPointer;
  typedef typename Superclass::PixelContainerConstPointer PixelContainerConstPointer;

  typedef typename Superclass::SpacingValueType SpacingValueType;
  typedef typename Superclass::PointValueType   PointValueType;
  typedef typename Superclass::IndexValueType   IndexValueType;
  typedef typename Superclass::SizeValueType    SizeValueType;
  typedef typename Superclass::OffsetValueType  OffsetValueType;

  /** ND versions of the index and sizetypes etc. */
  typedef typename Superclass::IndexType   IndexType;
  typedef typename Superclass::SizeType    SizeType;
  typedef typename Superclass::SpacingType SpacingType;
  typedef typename Superclass::PointType   PointType;
  typedef typename Superclass::OffsetType  OffsetType;

  /** Typedefs dependent on the dimension */
  typedef Image< TPixel, VDimension >  ImageType;
  typedef typename ImageType::Pointer  ImagePointer;
  typedef ImageFileWriter< ImageType > WriterType;
  typedef typename WriterType::Pointer WriterPointer;
  typedef ImageFileReader< ImageType > ReaderType;
  typedef typename ReaderType::Pointer ReaderPointer;

  /** Original, itk, versions of the index and sizetypes etc. */
  typedef typename ImageType::IndexType   IndexTypeD;
  typedef typename ImageType::SizeType    SizeTypeD;
  typedef typename ImageType::SpacingType SpacingTypeD;
  typedef typename ImageType::PointType   PointTypeD;
  typedef typename ImageType::OffsetType  OffsetTypeD;

  virtual void SetRegions( SizeType size );

  virtual void SetRequestedRegion( DataObject * data );

  virtual void Allocate( void );

  virtual void Initialize( void );

  virtual void FillBuffer( const TPixel & value );

  virtual void SetPixel( const IndexType & index, const TPixel & value );

  virtual const TPixel & GetPixel( const IndexType & index ) const;

  virtual TPixel & GetPixel( const IndexType & index );

  virtual TPixel * GetBufferPointer();

  virtual const TPixel * GetBufferPointer() const;

  virtual PixelContainer * GetPixelContainer();

  virtual const PixelContainer * GetPixelContainer() const;

  virtual void SetPixelContainer( PixelContainer * container );

  virtual AccessorType GetPixelAccessor( void );

  virtual const AccessorType GetPixelAccessor( void ) const;

  virtual void SetSpacing( const SpacingType & spacing );

  virtual void SetOrigin( const PointType & origin );

  virtual SpacingType GetSpacing( void );

  virtual PointType GetOrigin( void );

  /** \todo Transform IndexToPoint methods. */

  virtual void CopyInformation( const DataObject * data );

  virtual const OffsetValueType * GetOffsetTable() const;

  virtual OffsetValueType ComputeOffset( const IndexType & ind ) const;

  virtual IndexType ComputeIndex( OffsetValueType offset ) const;

  /** Extra functions for NDImage. */

  /** Get the Dimension.*/
  virtual unsigned int ImageDimension( void );

  virtual unsigned int GetImageDimension( void );

  /** Get the actual image */
  itkGetObjectMacro( Image, DataObject );
  itkGetObjectMacro( Writer, ProcessObject );
  itkGetObjectMacro( Reader, ProcessObject );

  /** Write the actual image to file. */
  virtual void Write( void );

  /** Read image data from file into the actual image */
  virtual void Read( void );

  /** Use New method to create a new actual image */
  virtual void CreateNewImage( void );

  virtual void SetImageIOWriter( ImageIOBase * _arg );

  virtual ImageIOBase * GetImageIOWriter( void );

  virtual void SetImageIOReader( ImageIOBase * _arg );

  virtual ImageIOBase * GetImageIOReader( void );

  /** Set/Get the Output/Input FileName */
  virtual void SetOutputFileName( const char * name );

  virtual void SetInputFileName( const char * name );

  virtual const char * GetOutputFileName( void );

  virtual const char * GetInputFileName( void );

protected:

  NDImageTemplate();
  virtual ~NDImageTemplate(){}

  //virtual void PrintSelf(std::ostream& os, Indent indent) const;

  ImagePointer  m_Image;
  WriterPointer m_Writer;
  ReaderPointer m_Reader;

  template< class TIn, class TOut >
  class ConvertToDynamicArray
  {
public:

    inline static TOut DO( const TIn & in )
    {
      TOut out( VDimension );

      for( unsigned int i = 0; i < VDimension; i++ )
      {
        out[ i ] = in[ i ];
      }
      return out;
    }


  };

  template< class TIn, class TOut >
  class ConvertToStaticArray
  {
public:

    inline static TOut DO( const TIn & in )
    {
      TOut out;

      for( unsigned int i = 0; i < VDimension; i++ )
      {
        out[ i ] = in[ i ];
      }
      return out;
    }


  };

private:

  NDImageTemplate( const Self & );  // purposely not implemented
  void operator=( const Self & );   // purposely not implemented

};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkNDImageTemplate.hxx"
#endif

#endif // end #ifndef __itkNDImageTemplate_h
