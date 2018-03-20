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

#ifndef __itkNDImageBase_h
#define __itkNDImageBase_h

#include "itkImage.h"
#include "itkArray.h"
#include <string>
#include "itkImageFileWriter.h"

namespace itk
{

/**
 * \class NDImageBase
 * \brief An image whose dimension can be specified at runtime.
 *
 * The NDImageBase class is needed for the FullSearch optimizer.
 * It allows run-time construction of an N-dimensional image. It
 * has most of the functionality of a normal itk::Image. \todo not all!
 * An internal writer and reader are included for convenience.
 *
 * The NewNDImage function defines the dimension of the image.
 * CreateNewImage creates an instance of an itk::Image, with
 * dimension as specified by NewNDImage.
 * Note: the NewNDImage does not return an itk::Image, but an
 * elx::NDImageTemplate.
 *
 * Suggested way of using this class:\n
 * NDImageBase<short> var1 = NDImageBase::NewNDImage(3);\n
 * var1->CreateNewImage();\n
 * The result is similar as:\n
 * itk::Image<short,3>::Pointer var1 = itk::Image<short,3>::New();\n
 * except that the actual itk::Image is stored as member variable
 * in the NDImageTemplate.
 *
 * \sa FullSearchOptimizer, NDImageTemplate
 * \ingroup Miscellaneous
 */

template< class TPixel >
class NDImageBase : public Object
{
public:

  /** Standard class typedefs.*/
  typedef NDImageBase                Self;
  typedef Object                     Superclass;
  typedef SmartPointer< Self >       Pointer;
  typedef SmartPointer< const Self > ConstPointer;

  /** Method for creation through the object factory. */
  // itkNewMacro( Self );
  // not declared, because instantiating an object of this
  // (abstract) type makes no sense.

  /** Run-time type information (and related methods). */
  itkTypeMacro( NDImageBase, Object );

  typedef DataObject              DataObjectType;
  typedef DataObjectType::Pointer DataObjectPointer;

  /** Type definitions like normal itkImages, independent of the dimension */
  typedef typename Image< TPixel, 2 >::PixelType                  PixelType;
  typedef typename Image< TPixel, 2 >::ValueType                  ValueType;
  typedef typename Image< TPixel, 2 >::InternalPixelType          InternalPixelType;
  typedef typename Image< TPixel, 2 >::AccessorType               AccessorType;
  typedef typename Image< TPixel, 2 >::PixelContainer             PixelContainer;
  typedef typename Image< TPixel, 2 >::PixelContainerPointer      PixelContainerPointer;
  typedef typename Image< TPixel, 2 >::PixelContainerConstPointer PixelContainerConstPointer;

  typedef typename ImageBase< 2 >::SpacingType Spacing2DType;
  typedef typename ImageBase< 2 >::PointType   Point2DType;

  typedef typename Spacing2DType::ValueType        SpacingValueType;
  typedef typename Point2DType::ValueType          PointValueType;
  typedef typename ImageBase< 2 >::IndexValueType  IndexValueType;
  typedef typename ImageBase< 2 >::SizeValueType   SizeValueType;
  typedef typename ImageBase< 2 >::OffsetValueType OffsetValueType;

  /** ND versions of the index and sizetypes. Unlike in
   * their counterparts in the itk::Image, their size
   * can be defined at runtime. The elx::NDImageTemplate
   * takes care of converting from/to these types to
   * to/from the corresponding types in itk::Image.*/
  typedef Array< IndexValueType >   IndexType;
  typedef Array< SizeValueType >    SizeType;
  typedef Array< SpacingValueType > SpacingType;
  typedef Array< PointValueType >   PointType;
  typedef Array< OffsetValueType >  OffsetType;
  /** \todo: extend to direction cosines; but not needed for now in elastix */

  /** Region typedef support. A region is used to specify a subset of an image. */

  //typedef typename Superclass::RegionType  RegionType;

  /** \todo an NDRegionType should first be declared, in the same way as NDImage
   * use SetRegions(size) for now. then knowlegde of the RegionType is not
   * necessary.
   * alternative: forget about the regiontype and add the functions
   * SetLargestPossibleRegion, SetRegions etc with arguments (index,size)
   * or maybe: ImageIORegion
   */

  //void SetRegions(RegionType region) = 0;
  virtual void SetRegions( SizeType size ) = 0;

  virtual void SetRequestedRegion( DataObject * data ) = 0;

  virtual void Allocate( void ) = 0;

  virtual void Initialize( void ) = 0;

  virtual void FillBuffer( const TPixel & value ) = 0;

  virtual void SetPixel( const IndexType & index, const TPixel & value ) = 0;

  virtual const TPixel & GetPixel( const IndexType & index ) const = 0;

  virtual TPixel & GetPixel( const IndexType & index ) = 0;

  TPixel & operator[]( const IndexType & index )
  { return this->GetPixel( index ); }
  const TPixel & operator[]( const IndexType & index ) const
  { return this->GetPixel( index ); }

  virtual TPixel * GetBufferPointer() = 0;

  virtual const TPixel * GetBufferPointer() const = 0;

  virtual PixelContainer * GetPixelContainer() = 0;

  virtual const PixelContainer * GetPixelContainer() const = 0;

  virtual void SetPixelContainer( PixelContainer * container ) = 0;

  virtual AccessorType GetPixelAccessor( void ) = 0;

  virtual const AccessorType GetPixelAccessor( void ) const = 0;

  virtual void SetSpacing( const SpacingType & spacing ) = 0;

  virtual void SetOrigin( const PointType & origin ) = 0;

  /* Get Spacing/Origin return copies; not a const &, like
   * itkImage; necessary because of the conversion to arrays */
  virtual SpacingType GetSpacing( void ) = 0;

  virtual PointType GetOrigin( void ) = 0;

  /** \todo Transform IndexToPoint methods. */

  virtual void CopyInformation( const DataObject * data ) = 0;

  virtual const OffsetValueType * GetOffsetTable() const = 0;

  virtual OffsetValueType ComputeOffset( const IndexType & ind ) const = 0;

  virtual IndexType ComputeIndex( OffsetValueType offset ) const = 0;

  /** Extra functions for NDImage. */

  /** Get the Dimension.*/
  virtual unsigned int ImageDimension( void ) = 0;

  virtual unsigned int GetImageDimension( void ) = 0;

  /** Get the actual image */
  virtual DataObject * GetImage( void ) = 0;

  virtual ProcessObject * GetWriter( void ) = 0;

  virtual ProcessObject * GetReader( void ) = 0;

  virtual void SetImageIOWriter( ImageIOBase * _arg ) = 0;

  virtual ImageIOBase * GetImageIOWriter( void ) = 0;

  virtual void SetImageIOReader( ImageIOBase * _arg ) = 0;

  virtual ImageIOBase * GetImageIOReader( void ) = 0;

  /** Write the actual image to file. */
  virtual void Write( void ) = 0;

  /** Read image data from file into the actual image */
  virtual void Read( void ) = 0;

  /** Use New method to create a new actual image */
  virtual void CreateNewImage( void ) = 0;

  /** Set/Get the Input/OutputFileName */
  virtual void SetOutputFileName( const char * ) = 0;

  virtual void SetInputFileName( const char * ) = 0;

  virtual const char * GetOutputFileName( void ) = 0;

  virtual const char * GetInputFileName( void ) = 0;

  static Pointer NewNDImage( unsigned int dim );

protected:

  NDImageBase(){}
  virtual ~NDImageBase(){}

  //virtual void PrintSelf(std::ostream& os, Indent indent) const = 0;

private:

  NDImageBase( const Self & );      // purposely not implemented
  void operator=( const Self & );   // purposely not implemented

};

} // end namespace itk

#include "itkNDImageTemplate.h"

namespace itk
{

template< class TPixel >
typename NDImageBase< TPixel >::Pointer
NDImageBase< TPixel >::NewNDImage( unsigned int dim )
{
  switch( dim )
  {
    case 1:
      return dynamic_cast< NDImageBase< TPixel > * >( NDImageTemplate< TPixel, 1 >::New().GetPointer() );
    case 2:
      return dynamic_cast< NDImageBase< TPixel > * >( NDImageTemplate< TPixel, 2 >::New().GetPointer() );
    case 3:
      return dynamic_cast< NDImageBase< TPixel > * >( NDImageTemplate< TPixel, 3 >::New().GetPointer() );
    case 4:
      return dynamic_cast< NDImageBase< TPixel > * >( NDImageTemplate< TPixel, 4 >::New().GetPointer() );
    case 5:
      return dynamic_cast< NDImageBase< TPixel > * >( NDImageTemplate< TPixel, 5 >::New().GetPointer() );
    //add here more dimensions if needed...
    // we could do this also with a recursive
    //template and a #define MAXDIM,
    // or something like that....
    default:
      return 0;
  }

}


} // end namespace itk

#endif // end #ifndef __itkNDImageBase_h
