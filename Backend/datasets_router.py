from fastapi import APIRouter, Depends, HTTPException, Response
from typing import List
import os
from database import get_db
import schemas
from datasets.dataset_loader import dataset_dirs, load_dataset_catalog, load_dataset_info, preview_image
import auth
from auth import get_current_user

router = APIRouter()

@router.get('/api/datasets', response_model=List[dict])
def list_datasets(current_user = Depends(get_current_user)):
    base = os.path.join(os.path.dirname(__file__), 'datasets')
    catalog = load_dataset_catalog(root=base)
    result = []
    for item in catalog:
        result.append({
            'name': item.get('name'),
            'classes': item.get('categories', []),
            'images': item.get('total_images', 0),
            'source_type': item.get('source_type', 'unknown'),
            'category_count': len(item.get('categories', [])),
            'categories': item.get('categories', []),
        })
    return result

@router.get('/api/datasets/info')
def dataset_info(name: str, current_user = Depends(get_current_user)):
    base = os.path.join(os.path.dirname(__file__), 'datasets')
    info = load_dataset_info(name, root=base)
    if not info.get('exists'):
        raise HTTPException(status_code=404, detail='Dataset not found')
    return info

@router.get('/api/datasets/statistics')
def dataset_statistics(name: str, current_user = Depends(get_current_user)):
    info = load_dataset_info(name, root=os.path.join(os.path.dirname(__file__), 'datasets'))
    if not info.get('exists'):
        raise HTTPException(status_code=404, detail='Dataset not found')
    return {
        'name': name,
        'total_images': info.get('total_images', 0),
        'class_counts': info.get('class_counts', {})
    }

@router.get('/api/datasets/preview')
def dataset_preview(name: str, class_name: str = None, index: int = 0, current_user = Depends(get_current_user)):
    data = preview_image(name, class_name, index, root=os.path.join(os.path.dirname(__file__), 'datasets'))
    if not data:
        raise HTTPException(status_code=404, detail='Preview not found')
    return Response(content=data, media_type='image/png')
